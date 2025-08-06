import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import shutil
from PIL import Image
import random
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import csv

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_SAVE_PATH = "binary_cnn_model.pth"  # Path to save the trained model weights

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
IMAGE_SIZE = 200 # All images will be resized to this (e.g., 64x64 pixels)

# Define transformations for the images
# 1. Resize: Ensures all images have the same dimensions.
# 2. ToTensor: Converts PIL Image or NumPy array to PyTorch tensor (HWC to CHW, scales to [0,1]).
# 3. Normalize: Standardizes pixel values (mean, std deviation for each channel).
#    These values (0.5, 0.5, 0.5) are common for images, but you might calculate
#    them from your specific dataset for better performance.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the dataset using ImageFolder
# ImageFolder expects data organized as:
# root/class_0/xxx.png
# root/class_1/yyy.png
# It automatically assigns labels based on folder names.
dataset = datasets.ImageFolder(root="trainingdata", transform=transform)

# Split the dataset into training and testing sets
total_size = len(dataset)
train_size = int(0.5 * total_size)
val_size = int(0.3 * total_size)
test_size = total_size - train_size - val_size

# First split: Separate out the test set
train_val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size + val_size, test_size])

# Second split: Separate train and validation from the remaining dataset
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])


# Create DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # No need to shuffle validation
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")
print(f"Classes: {dataset.classes}") # Should now reflect 'Atypical' and 'Typical'

# --- 3. Model Definition (Simple CNN) ---
class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        # Convolutional Layer 1
        # Input: (Batch_size, 3, IMAGE_SIZE, IMAGE_SIZE) -> 3 channels for RGB
        # Output: (Batch_size, 16, IMAGE_SIZE/2, IMAGE_SIZE/2) after MaxPool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        # Input: (Batch_size, 16, IMAGE_SIZE/2, IMAGE_SIZE/2)
        # Output: (Batch_size, 32, IMAGE_SIZE/4, IMAGE_SIZE/4) after MaxPool
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer
        # Calculate input features for the first linear layer.
        # After two pooling layers with stride 2, the image dimensions are reduced by 2^2 = 4.
        # So, IMAGE_SIZE / 4 is the dimension.
        # 32 is the number of output channels from the last conv layer.
        self.fc_input_features = 32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4)
        self.fc = nn.Linear(self.fc_input_features, 1) # Output 1 for binary classification

    def forward(self, x):
        # Pass through Conv1 -> ReLU -> Pool1
        x = self.pool1(self.relu1(self.conv1(x)))
        # Pass through Conv2 -> ReLU -> Pool2
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten the output for the fully connected layer
        # x.view(-1, self.fc_input_features) reshapes the tensor.
        # -1 means infer the batch size, self.fc_input_features is the total features per image.
        x = x.view(-1, self.fc_input_features)

        # Pass through the fully connected layer
        # No sigmoid here, as BCEWithLogitsLoss handles it internally for numerical stability.
        x = self.fc(x)
        return x

# Instantiate the model and move it to the device
model = BinaryCNN().to(device)
print("\nModel Architecture:")
print(model)

# --- 4. Loss Function and Optimizer ---
# BCEWithLogitsLoss combines Sigmoid and Binary Cross Entropy Loss.
# It's more numerically stable than using Sigmoid + BCELoss separately.
criterion = nn.BCEWithLogitsLoss()

# Adam optimizer is a good general-purpose optimizer.
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training the Model ---
print("\nStarting Training...")
for epoch in range(NUM_EPOCHS):
    model.train() # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

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

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions * 100
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

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

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct_predictions / val_total_predictions * 100
    print(f"          Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

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

# Calculate F1-score, Precision, and Recall
# Specify pos_label for binary classification if needed, based on your class definition
f1 = f1_score(all_labels_int, all_predictions_int, pos_label=pos_label_idx)
precision = precision_score(all_labels_int, all_predictions_int, pos_label=pos_label_idx)
recall = recall_score(all_labels_int, all_predictions_int, pos_label=pos_label_idx)

print(f"F1-Score (for {dataset.classes[pos_label_idx]}): {f1:.4f}")
print(f"Precision (for {dataset.classes[pos_label_idx]}): {precision:.4f}")
print(f"Recall (for {dataset.classes[pos_label_idx]}): {recall:.4f}")

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
