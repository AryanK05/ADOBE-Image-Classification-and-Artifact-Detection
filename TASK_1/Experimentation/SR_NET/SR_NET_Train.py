import os
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader


def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, num_epochs: int = 25, device: torch.device = torch.device("cpu")) -> None:
    """
    Train the model for a specified number of epochs.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader object for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device for training (CPU or GPU).
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1), accuracy=f"{100 * correct_train / total_train:.2f}%")

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Save the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(save_dir, f"srdnet_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} to {save_path}")

def validate(model: nn.Module, val_loader: DataLoader, device: torch.device = torch.device("cpu")) -> None:
    """
    Validate the model on validation data.

    Args:
        model (nn.Module): The model to be validated.
        val_loader (DataLoader): DataLoader object for validation data.
        device (torch.device): Device for validation (CPU or GPU).
    """
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", unit="batch") as vepoch:
            for images, labels in vepoch:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                vepoch.set_postfix(accuracy=f"{100 * correct_val / total_val:.2f}%")

    val_accuracy = 100 * correct_val / total_val
    print(f"Validation Accuracy: {val_accuracy:.2f}%")