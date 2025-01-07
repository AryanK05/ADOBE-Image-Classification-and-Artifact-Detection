import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

def train_and_validate(model, train_loader, val_loader, epochs, save_path="model_epoch_{}.pt"):
    """
    Trains and validates the model for a specified number of epochs.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        epochs (int): Number of epochs to train the model.
        save_path (str): Path to save the model weights.

    """
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training", unit="batch") as t:
            for images, labels in t:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                t.set_postfix(loss=train_loss / total, accuracy=train_correct / total)

        # Validate the model
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation", unit="batch") as t:
                for images, labels in t:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    t.set_postfix(loss=val_loss / val_total, accuracy=val_correct / val_total)

        # Save the model at the end of each epoch
        torch.save(model.state_dict(), save_path.format(epoch + 1))