import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from clean_dataset_224 import train_loader,val_loader


# Define the DINO Classifier model
class DinoClassifier(nn.Module):
    """
    A classifier built on top of the DINOv2 feature extractor.

    Args:
        feature_extractor (nn.Module): Pretrained DINOv2 model as feature extractor.
        output_dim (int): Number of classes for the classification task.
    """

    def __init__(self, feature_extractor, output_dim=2):
        super(DinoClassifier, self).__init__()
        self.feature_extractor = feature_extractor

        # Freeze DINO model parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Add a classification head
        self.classifier = nn.Linear(self.feature_extractor.config.hidden_size, output_dim)

    def forward(self, images):
        """
        Forward pass for the model.

        Args:
            images (torch.Tensor): Input batch of images.

        Returns:
            logits (torch.Tensor): Output logits for each class.
        """
        inputs = processor(images=images, return_tensors="pt").to(images.device)
        with torch.no_grad():
            outputs = self.feature_extractor(**inputs)
            features = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(features)
        return logits


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Train and validate the model.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train.
    """
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss, correct, total = 0, 0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False) as train_bar:
            for images, labels in train_bar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                train_bar.set_postfix(loss=loss.item())

        train_accuracy = correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")

        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False) as val_bar:
                for images, labels in val_bar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the model at regular intervals
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(save_dir, f"dino_classifier_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} to {save_path}")

        model.train()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the DINOv2 processor and model
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')
dino_classifier = DinoClassifier(model).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(dino_classifier.classifier.parameters(), lr=1e-4)

# Directory to save model checkpoints
save_dir = "./model_checkpoints"
os.makedirs(save_dir, exist_ok=True)
train_model(dino_classifier, train_loader, val_loader, criterion, optimizer, num_epochs=10)
