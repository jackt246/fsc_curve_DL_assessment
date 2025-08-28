import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
from emdb.client import EMDB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from datetime import datetime, timedelta, timezone

model_path = 'typicality_model.pth'
class_names = ['1_Atypical', '0_Typical']

def previous_wednesday():
    today = datetime.now(timezone.utc).date()
    # weekday(): Monday=0, Sunday=6. Wednesday=2
    offset = (today.weekday() - 2) % 7
    if offset == 0:
        offset = 7  # if today is Wednesday, go to the previous one
    last_wed = today - timedelta(days=offset)
    return last_wed

# initialise the EMDB client and get entry
client = EMDB()

# Search for the previous release
last_wed = previous_wednesday()
query_date = last_wed.strftime("%Y-%-m-%-dT00:00:00Z")
results = client.search(f'release_date:"{query_date}" AND database:EMDB')

def process_entry(entry):
    try:
        validation = entry.get_validation()
        result = {
            'entry': entry.id,
            'calculated_fsc_y': validation.plots.fsc.fsc,
            'corrected_fsc_y': validation.plots.fsc.fsc_corrected,
            'phase_randomised_y': validation.plots.fsc.phaserandomization,
            'fsc_x': validation.plots.fsc.level,
            'resolution': validation.plots.fsc.resolution,
            '1/resolution': 1 / validation.plots.fsc.resolution
        }
        return result
    except TimeoutError as e:
        print(f"Timeout occurred for entry {entry.id}: {e}")
        return None
    except Exception as e:
        print(f"Error processing entry {entry.id}: {e}")
        return None

list_for_csv = []

for entry in results:
    result = process_entry(entry)
    try:
        fsc_x = result['fsc_x']
        calculated_fsc_y = result['calculated_fsc_y']
        corrected_fsc_y = result['corrected_fsc_y']
        phase_randomised_y = result['phase_randomised_y']
    except (KeyError, TypeError) as e:
        print(f"Skipping entry {entry.id} due to missing data: {e}")
        continue


    # Create plotting DataFrame just for seaborn
    plot_df = pd.DataFrame({
        'fsc_x': fsc_x,
        'Calculated Unmasked FSC': calculated_fsc_y,
        'Corrected Masked FSC': corrected_fsc_y,
        'Phase Randomised FSC': phase_randomised_y
    }).melt(id_vars='fsc_x', var_name='Curve', value_name='Correlation')

    # Plot
    plt.figure(figsize=(6,4))
    sns.lineplot(data=plot_df, x='fsc_x', y='Correlation', hue='Curve')
    plt.title(result['entry'])
    plt.xlabel('Spatial Frequency (1/Å)')
    plt.ylabel('Correlation')
    plt.axvline(x=result['1/resolution'], color='red', linestyle='--', label='resolution')
    plt.grid(True)
    plt.legend()
    plt.savefig('emdb_entry.png')


    # Match the architecture you trained with
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=1)  # Binary classification

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # match training size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open('emdb_entry.png').convert("RGB")     # Load and ensure RGB
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():  # Disable gradient calc for speed
        output = model(img_t)
        prob = torch.sigmoid(output)  # For binary classification
        pred = (prob > 0.5).int().item()

    predicted_class = class_names[pred]
    confidence = prob.item() if pred == 1 else 1 - prob.item()
    list_for_csv.append({
        "entry": entry.id,
        "prediction": predicted_class,
        "confidence": round(confidence, 4)
    })

df = pd.DataFrame(list_for_csv)
df.to_csv("predictions.csv", index=False)