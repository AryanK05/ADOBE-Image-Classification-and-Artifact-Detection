import os
import torch
from torch import nn, optim
from Facebook_DEiT_model import initialize_model
from Facebook_DEiT_Train import train_model
from Facebook_DEiT_Test import test_model
from clean_dataset_224 import train_loader, val_loader
from Test_Dataset_pipeline_224 import test_loader

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters and setup
    batch_size = 32
    learning_rate = 5e-5
    epochs = 40
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # Model, optimizer, and loss
    model = initialize_model(num_classes=2, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, epochs, checkpoint_dir, device)

    # Load the best checkpoint for testing
    model_path = "/content/deit_finetuned_general_final.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Test the model
    print("Starting testing...")
    test_model(model, test_loader, device)
