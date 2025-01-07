import torch
from Res_Net_train import train_model
from Res_Net_Test import test_model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(device, num_epochs=10)
    checkpoint_path = ""
    test_model(device, checkpoint_path)
