import torch
from tqdm import tqdm
import time

def test_model(model, test_loader, device):
    """
    Evaluates the ViT model on the test dataset.
    Args:
        model (torch.nn.Module): Fine-tuned ViT model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to use for evaluation.
    """
    model.eval()
    correct, total = 0, 0
    total_inference_time = 0.0

    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                start_time = time.time()
                outputs = model(pixel_values=images).logits
                end_time = time.time()
                total_inference_time += (end_time - start_time)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tepoch.set_postfix(accuracy=f"{100 * correct / total:.2f}%")

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
