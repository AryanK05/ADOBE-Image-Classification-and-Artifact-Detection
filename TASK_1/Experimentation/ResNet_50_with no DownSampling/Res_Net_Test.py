import torch
from tqdm import tqdm
from Res_Net_model import resnet50_custom_model
from Test_Dataset_pipeline_224 import test_loader

def test_model(device, checkpoint_path="trained_model.pth"):
    model = resnet50_custom_model(device, num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()

    correct, total = 0, 0
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tepoch.set_postfix(accuracy=f"{100 * correct / total:.2f}%")

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
