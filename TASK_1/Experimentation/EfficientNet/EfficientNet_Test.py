import torch
from tqdm import tqdm
import time

# Testing Section
def test_model(model, test_loader, criterion, device):
    """
    Test the model and evaluate metrics.
    Args:
        model (nn.Module): Model to test.
        test_loader (DataLoader): DataLoader for testing data.
        criterion: Loss function.
        device: Device to use (CPU/GPU).
    """
    model.eval()
    test_loss, correct, total, total_inference_time = 0.0, 0, 0, 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_inference_time = total_inference_time / total
    epoch_loss = test_loss / len(test_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, avg_inference_time