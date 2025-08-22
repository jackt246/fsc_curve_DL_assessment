import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

model_path = 'typicality_model.pth'
image_path = 'EMD-19436.png'
class_names = ['Atypical', 'Typical']


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


img = Image.open(image_path).convert("RGB")
img_t = transform(img).unsqueeze(0)  # Add batch dimension

with torch.no_grad():  # Disable gradient calc for speed
    output = model(img_t)
    prob = torch.sigmoid(output)  # For binary classification
    pred = (prob > 0.5).int().item()

predicted_class = class_names[pred]
confidence = prob.item() if pred == 1 else 1 - prob.item()
print(f"Prediction: {predicted_class} | Confidence: {confidence:.4f}")