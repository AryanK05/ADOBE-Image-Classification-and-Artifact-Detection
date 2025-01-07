import torch
import os
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from SR_NET_model import Srnet
from SR_NET_Train import train, validate
from SR_NET_Test import test_model
from clean_dataset import train_loader, val_loader
from Test_Dataset_pipeline import test_loader


if __name__ == "__main__":
    # Define parameters and setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # Define model, criterion, optimizer
    model = Srnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs=25, device=device)

    # Validate the model
    validate(model, val_loader, device=device)

    # Test the model
    test_model(model, "/content/dino_classifier_epoch_12.pth", test_loader, device=device)