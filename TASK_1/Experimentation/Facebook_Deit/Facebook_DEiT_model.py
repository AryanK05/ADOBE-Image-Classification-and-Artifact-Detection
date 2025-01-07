import os
import time
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, ViTForImageClassification, DeiTForImageClassificationWithTeacher
from torch import nn, optim


# ============================ Model Architecture ============================


def initialize_model(num_classes=2):
    """
    Initializes the ViT model for fine-tuning.
    Args:
        num_classes (int): Number of output classes.
    Returns:
        torch.nn.Module: Initialized model.
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
    model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    return model.to(device)




