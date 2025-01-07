import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_model(classifier, train_loader, val_loader, criterion, optimizer, device, checkpoint_dir, num_epochs=10):
    """
    Train the ViT binary classifier model.

    Args:
        classifier (nn.Module): The model to be trained.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        num_epochs (int): The number of epochs for training.
    """
    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0.0
        y_true_train, y_pred_train = [], []

        print(f"Epoch [{epoch + 1}/{num_epochs}] Training")
        for batch in tqdm(train_loader, desc="Training Batches"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())

        train_accuracy = accuracy_score(y_true_train, y_pred_train)
        print(f"Training Loss: {train_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}")

        # Validation Loop
        classifier.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []

        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation Batches"):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = classifier(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        val_accuracy = accuracy_score(y_true_val, y_pred_val)
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(classifier.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    print("Training completed.")
