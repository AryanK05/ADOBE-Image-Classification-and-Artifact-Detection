import os
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from sklearn.metrics import classification_report

def test_model(model, checkpoint_path, test_loader, batch_size=32, device=None):
    """
    Function to test a model on a dataset and print the results.

    Args:
        model (torch.nn.Module): The model to be tested.
        checkpoint_path (str): Path to the model checkpoint file.
        test_dataset_path (str): Path to the test dataset.
        batch_size (int): Batch size for the DataLoader. Default is 32.
        device (torch.device or None): Device to run the model on (GPU or CPU). If None, uses available device.

    Returns:
        dict: A dictionary containing accuracy, total inference time, average inference time per sample,
              and the classification report.
    """

    # Device configuration (default is GPU if available, otherwise CPU)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Switch model to evaluation mode
    model.eval()

    # Variables for tracking performance
    correct, total = 0, 0
    total_inference_time = 0.0
    all_labels = []
    all_predictions = []

    # Test loop
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                # Measure inference time
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()

                # Update total inference time
                total_inference_time += (end_time - start_time)

                # Predictions and accuracy calculations
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tepoch.set_postfix(accuracy=f"{100 * correct / total:.2f}%")

    # Calculate final metrics
    accuracy = 100 * correct / total
    avg_inference_time_per_sample = total_inference_time / len(test_loader.dataset)

    # Generate classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=test_loader.dataset.classes,
        digits=4
    )

    # Print results
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"Average Inference Time per Sample: {avg_inference_time_per_sample:.6f} seconds")
    print("\nClassification Report:\n")
    print(report)