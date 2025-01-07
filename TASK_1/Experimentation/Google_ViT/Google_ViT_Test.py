from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch
import torch.nn as nn
def test_model_with_metrics(classifier, test_loader, device):
    """
    Evaluate the model on the test dataset and compute relevant metrics.

    Args:
        classifier (nn.Module): The trained model.
        test_loader (DataLoader): The test data loader.
        device (str): The device to run the model on (either 'cuda' or 'cpu').

    Returns:
        tuple: Test loss, accuracy, precision, recall, and F1 score.
    """
    classifier.eval()
    test_loss = 0.0
    y_true_test, y_pred_test = [], []

    # Define loss function (consistent with training)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Batches"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = classifier(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(preds.cpu().numpy())

    test_accuracy = accuracy_score(y_true_test, y_pred_test)
    precision = precision_score(y_true_test, y_pred_test, average='weighted')
    recall = recall_score(y_true_test, y_pred_test, average='weighted')
    f1 = f1_score(y_true_test, y_pred_test, average='weighted')

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return test_loss, test_accuracy, precision, recall, f1