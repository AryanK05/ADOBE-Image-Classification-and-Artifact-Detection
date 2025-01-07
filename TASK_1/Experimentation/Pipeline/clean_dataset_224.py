import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from PIL import Image, ImageEnhance
import numpy as np

# Function to add Gaussian noise
def add_gaussian_noise(img, mean=0, std=0.05):
    np_img = np.array(img)
    noise = np.random.normal(mean, std, np_img.shape)
    noisy_img = np_img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)  # Clip to valid range
    return Image.fromarray(noisy_img)

train_transform = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.LANCZOS),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),  # Random rotation up to 15 degrees
    transforms.RandomAffine(15, scale=(0.8, 1.2), shear=10),  # Random affine transformation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Color jitter
    transforms.Lambda(lambda img: add_gaussian_noise(img, mean=0, std=0.05)),  # Gaussian noise
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root='/content/archive/PixART dataset', transform=train_transform)

# Define function to get class indices
def get_class_indices(dataset, class_name):
    class_idx = dataset.class_to_idx[class_name]
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
    return indices

# Getting indices for the "REAL" and "FAKE" classes in the full dataset
real_indices = get_class_indices(full_dataset, 'REAL')
fake_indices = get_class_indices(full_dataset, 'FAKE')

random.seed(42)

real_val_selected = random.sample(real_indices, 5000)
fake_val_selected = random.sample(fake_indices, 5000)


val_selected_indices = real_val_selected + fake_val_selected

# Create the validation subset with no augmentation
val_subset = Subset(datasets.ImageFolder(root='/content/archive/PixART dataset', transform=val_transform), val_selected_indices)

# Use the remaining indices for the training set with augmentation
train_indices = [i for i in range(len(full_dataset)) if i not in val_selected_indices]
train_subset = Subset(full_dataset, train_indices)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

print(f"Train dataset size: {len(train_subset)}")
print(f"Validation dataset size: {len(val_subset)}")
