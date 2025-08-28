import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import csv
from torchvision import models
from collections import Counter
import os
# Fix SSL certificate issue for torchvision model downloads on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Set the device to GPU if available, otherwise CPU (with MPS support for Apple Silicon)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

MODEL_SAVE_PATH = "../inference/typicality_model.pth"  # Path to save the trained model weights

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 25

# Load pretrained EfficientNetV2 with ImageNet1k_V1 weights
weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
model = models.efficientnet_v2_s(weights=weights)

# Separate transforms for training and validation/test
# Compose training transform with augmentations, then tensor/normalize with weights' mean and std
train_transform = transforms.Compose([
#    transforms.RandomHorizontalFlip(),
#    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
])
test_transform = weights.transforms()

# This could probably be one line but I am keeping it explicit for clarity and because im too lazy to refactor it
# on friday afternoon.
base_dataset = datasets.ImageFolder(root="trainingdata/Typicality")
dataset = base_dataset  # Keep reference for later use (e.g., image_path lookups)

# Split the dataset
total_size = len(base_dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_val_dataset, test_dataset = torch.utils.data.random_split(base_dataset, [train_size + val_size, test_size])
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

# Apply transforms individually
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform

class_counts = Counter([label for _, label in dataset.samples])
print(f"Class counts: {class_counts}")
# Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance
# pos_weight should be minority_class / majority_class to upweight the minority class
pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=device)

# But we need to determine which is actually the minority class
if class_counts[0] < class_counts[1]:
    # Class 0 is minority, Class 1 is majority
    pos_weight = torch.tensor([class_counts[1] / class_counts[0]], device=device)
    print(f"Class 0 ({dataset.classes[0]}) is minority with {class_counts[0]} samples")
    print(f"Class 1 ({dataset.classes[1]}) is majority with {class_counts[1]} samples")
else:
    # Class 1 is minority, Class 0 is majority
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=device)
    print(f"Class 1 ({dataset.classes[1]}) is minority with {class_counts[1]} samples")
    print(f"Class 0 ({dataset.classes[0]}) is majority with {class_counts[0]} samples")

print(f"Using pos_weight: {pos_weight.item():.4f}")

# Create DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")



# Modify the classifier for binary classification
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 1)

# Send to device
model = model.to(device)
print("\nModel Architecture:")
print(model)

# --- 4. Loss Function and Optimizer ---
# BCEWithLogitsLoss combines Sigmoid and Binary Cross Entropy Loss.
# It's more numerically stable than using Sigmoid + BCELoss separately.
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

# --- 5. Training the Model ---
print("\nStarting Training...")
for epoch in range(NUM_EPOCHS):
    model.train() # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    all_train_preds = []
    all_train_labels = []
    all_val_preds = []
    all_val_labels = []

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1) # Labels need to be float and shape (batch_size, 1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        # Apply sigmoid to outputs and threshold at 0.5 to get binary predictions
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Store for precision calculation
        all_train_preds.extend(predicted.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())


    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions * 100
    epoch_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
    epoch_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
    epoch_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
    print(f"Training: Epoch [{epoch + 1}/{NUM_EPOCHS}], "
          f"Loss: {epoch_loss:.4f}, "
          f"Accuracy: {epoch_accuracy:.2f}%, "
          f"Precision: {epoch_precision:.4f}, "
          f"Recall: {epoch_recall:.4f}, "
          f"F1 Score: {epoch_f1:.4f}")

    # --- Validation Phase ---
    model.eval() # Set the model to evaluation mode for validation
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            val_total_predictions += labels.size(0)
            val_correct_predictions += (predicted == labels).sum().item()

            # Store for precision calculation
            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct_predictions / val_total_predictions * 100
    val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0)
    val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0)
    val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
    print(f"Validation: Epoch [{epoch + 1}/{NUM_EPOCHS}], "
          f"Loss: {val_loss:.4f}, "
          f"Accuracy: {val_accuracy:.2f}%, "
          f"Precision: {val_precision:.4f}, "
          f"Recall: {val_recall:.4f}, "
          f"F1 Score: {val_f1:.4f}")

    if val_f1 > 0.98:
        print(f"Stopping early as F1 score reached {val_f1:.4f}")
        continue

print("Training Complete!")

# --- 6. Evaluating the Model and Calculating Metrics on the Test Set ---
print("\nStarting Evaluation and Metric Calculation on Test Set...")
model.eval() # Set the model to evaluation mode (disables dropout, batchnorm updates)
all_labels = []
all_predictions = []
all_probabilities = [] # To store probabilities for CSV
test_results_for_csv = [] # List to store results for CSV

with torch.no_grad(): # Disable gradient calculation during evaluation
    for i in range(len(test_dataset)):
        # Get individual sample from test_dataset
        image, true_label_idx = test_dataset[i]
        image_path = dataset.samples[test_dataset.indices[i]][0] # Get original image path

        # Add batch dimension and move to device
        inputs = image.unsqueeze(0).to(device)
        labels = torch.tensor([true_label_idx]).float().unsqueeze(1).to(device)

        outputs = model(inputs)
        probability = torch.sigmoid(outputs).item() # Get probability of the positive class
        # Fix: Convert boolean result of comparison directly to float
        predicted_label_idx = float(probability > 0.5)

        all_labels.append(true_label_idx)
        all_predictions.append(predicted_label_idx)
        all_probabilities.append(probability)

        # Determine if correct
        is_correct = "Correct" if predicted_label_idx == true_label_idx else "Incorrect"

        # Store results for CSV
        test_results_for_csv.append({
            "Image Path": image_path,
            "True Label": dataset.classes[true_label_idx],
            "Predicted Label": dataset.classes[int(predicted_label_idx)],
            "Predicted Probability (Typical)": f"{probability:.4f}", # Assuming 'Typical' is class 1
            "Result": is_correct
        })


# Convert lists to numpy arrays
all_labels = np.array(all_labels).flatten()
all_predictions = np.array(all_predictions).flatten()

# Calculate overall accuracy
accuracy = np.sum(all_predictions == all_labels) / len(all_labels) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Determine the index for the 'Typical' class, assuming it's the positive class
# ImageFolder assigns labels alphabetically, so 'Atypical' (0), 'Typical' (1)
pos_label_idx = dataset.class_to_idx.get('Typical', 1) # Default to 1 if not found, assuming Typical is the positive class
# Ensure that all_labels and all_predictions are integers for sklearn metrics
all_labels_int = all_labels.astype(int)
all_predictions_int = all_predictions.astype(int)

# Calculate per-class F1, Precision, and Recall
f1_scores = f1_score(all_labels_int, all_predictions_int, average=None, labels=[0,1])
precision_scores = precision_score(all_labels_int, all_predictions_int, average=None, labels=[0,1])
recall_scores = recall_score(all_labels_int, all_predictions_int, average=None, labels=[0,1])

print(f"F1-Score (for {dataset.classes[0]}): {f1_scores[0]:.4f}")
print(f"Precision (for {dataset.classes[0]}): {precision_scores[0]:.4f}")
print(f"Recall (for {dataset.classes[0]}): {recall_scores[0]:.4f}")

print(f"F1-Score (for {dataset.classes[1]}): {f1_scores[1]:.4f}")
print(f"Precision (for {dataset.classes[1]}): {precision_scores[1]:.4f}")
print(f"Recall (for {dataset.classes[1]}): {recall_scores[1]:.4f}")

# Calculate Confusion Matrix
cm = confusion_matrix(all_labels_int, all_predictions_int)
print("\nConfusion Matrix:")
print(cm)

# --- 7. Plotting the Confusion Matrix ---
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Get class names from the dataset
class_names = dataset.classes # e.g., ['Atypical', 'Typical']

# Plot the non-normalized confusion matrix
plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')

# Plot the normalized confusion matrix (optional)
# plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized Confusion Matrix')

# --- 8. Outputting Classification Results to CSV ---
csv_filename = "classification_results.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ["Image Path", "True Label", "Predicted Label", "Predicted Probability (Typical)", "Result"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in test_results_for_csv:
        writer.writerow(row)
print(f"\nClassification results saved to {csv_filename}")


# --- 9. Saving the Model Weights ---
print(f"\nSaving model weights to {MODEL_SAVE_PATH}...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model weights saved successfully.")

# --- Optional: Clean up dummy data ---
# shutil.rmtree("data")
# print("Cleaned up dummy data.")
