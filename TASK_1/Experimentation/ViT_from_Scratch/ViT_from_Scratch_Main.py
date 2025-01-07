import torch
from ViT_from_Scratch_model import CvT
from ViT_from_Scratch_Train import train_and_validate
from clean_dataset import train_loader, val_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CvT(embed_dim=384, num_class=10)
    train_and_validate(model, train_loader, val_loader, epochs=20)