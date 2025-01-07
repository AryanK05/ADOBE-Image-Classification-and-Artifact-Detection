import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Res_Net_model import resnet50_custom_model
from clean_dataset_224 import train_loader, val_loader

def train_model(device, num_epochs=10):
    model = resnet50_custom_model(device, num_classes=2)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_predictions, total_predictions = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_predictions / total_predictions
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        model.eval()
        running_val_loss, correct_val_predictions, total_val_predictions = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val_predictions += (predicted == labels).sum().item()
                total_val_predictions += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val_predictions / total_val_predictions
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        print('-' * 50)

    torch.save(model.module.state_dict(), "trained_model.pth")
