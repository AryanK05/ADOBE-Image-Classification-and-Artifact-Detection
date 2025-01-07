import os
import torch
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, epochs, checkpoint_dir, device):
    """
    Fine-tunes the ViT model on the training dataset.
    Args:
        model (torch.nn.Module): Pre-trained ViT model.
        train_loader (DataLoader): Training data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Number of training epochs.
        checkpoint_dir (str): Directory to save model checkpoints.
        device (torch.device): Device to use for training.
    """
    model.train()
    for epoch in range(epochs):
        train_loss, train_correct = 0, 0
        total_train_samples = 0
        tqdm_train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]", leave=False)
        for images, labels in tqdm_train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train_samples += labels.size(0)
            tqdm_train_loader.set_postfix(loss=(train_loss / total_train_samples))

        train_acc = train_correct / total_train_samples
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}")

        if (epoch + 1) % 4 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss / len(train_loader),
                'train_accuracy': train_acc,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
