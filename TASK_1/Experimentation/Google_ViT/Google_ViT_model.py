import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTImageProcessor, ViTModel
import os

# ============================ Model Architecture ============================

class ViTBinaryClassifier(nn.Module):
    """
    A binary classifier using the Vision Transformer (ViT) model with a fully connected layer on the [CLS] token.

    Args:
        vit_model (ViTModel): The pre-trained ViT model to be used as a backbone.
        num_classes (int): The number of classes for classification (default is 2).
    """
    def __init__(self, vit_model, num_classes=2):
        super(ViTBinaryClassifier, self).__init__()
        self.vit = vit_model
        self.fc = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the ViT model.

        Args:
            x (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The output class logits.
        """
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.pooler_output
        return self.fc(cls_token)

