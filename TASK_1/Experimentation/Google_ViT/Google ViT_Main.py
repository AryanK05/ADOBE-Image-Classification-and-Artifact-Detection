import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTImageProcessor, ViTModel
import os
from Google_ViT_model import ViTBinaryClassifier
from Google_ViT_Train import train_model
from Google_ViT_Test import test_model_with_metrics
from clean_dataset_224 import train_loader, val_loader
from Test_Dataset_pipeline_224 import test_loader

# ============================ Main Execution ============================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the pretrained ViT model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Freeze ViT model parameters (optional)
    for param in model.parameters():
        param.requires_grad = False

    # Instantiate the classifier model and move it to the device
    classifier = ViTBinaryClassifier(model, num_classes=2).to(device)

    # Use DataParallel for multi-GPU support if available
    if torch.cuda.device_count() > 1:
        classifier = nn.DataParallel(classifier)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

    # Create checkpoint directory
    checkpoint_dir = './model_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    num_epochs = 10
    train_model(classifier, train_loader, val_loader,  criterion, optimizer, device, checkpoint_dir, num_epochs)

    # Load the best model checkpoint for evaluation
    best_model_path = '/kaggle/working/model_checkpoints/model_epoch_10.pth'  # Adjust as necessary
    classifier.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded model checkpoint from {best_model_path}")

    # Evaluate the model on the test dataset
    test_loss, test_accuracy, precision, recall, f1 = test_model_with_metrics(classifier, test_loader, device)
