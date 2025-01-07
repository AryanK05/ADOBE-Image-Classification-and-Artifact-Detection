import torch
import torch.nn as nn
import torch.optim as optim
from clean_dataset import train_loader, val_loader  # Import pre-defined DataLoader objects
from EfficientNet_model import EfficientNet  # Import your model class
from EfficientNet_Training import train_model  # Import the training function
from EfficientNet_Test import test_model  # Import the testing function

if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = EfficientNet(version="b0").to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()  # Use Binary Cross Entropy Loss for binary classification

    # Training phase
    print("Starting training...")
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device=device)
    print(f"Training completed. Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Load pre-trained model
    model_path = "/content/efficientNetb0_smalldata (1).pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Testing phase
    print("Starting testing...")
    test_loss, test_acc, test_time = test_model(model, val_loader, criterion, device=device)
    print(f"Testing completed. Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # Summary
    print("Process completed.")
