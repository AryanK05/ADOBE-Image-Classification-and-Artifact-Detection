from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from PIL import Image

# Define transformations for training dataset
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class AugmentationTransform:
    def __init__(self):
        self.resize = transforms.Resize((224, 224), interpolation=Image.LANCZOS)
        self.random_crop = transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33))
        self.random_rotation = transforms.RandomRotation(degrees=30)
        self.color_jitter = transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.01)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def add_gaussian_noise(self, img_tensor, mean=0.0, std=0.1):
        noise = torch.randn_like(img_tensor) * std + mean
        return img_tensor + noise

    def add_salt_and_pepper_noise(self, img_tensor, prob=0.01):
        img_array = img_tensor.permute(1, 2, 0).numpy()  # Convert to HWC for easier manipulation
        h, w, c = img_array.shape
        num_pixels = h * w

        # Add salt noise
        num_salt = int(prob * num_pixels / 2)
        for _ in range(num_salt):
            x, y = np.random.randint(0, h), np.random.randint(0, w)
            img_array[x, y] = 1.0

        # Add pepper noise
        num_pepper = int(prob * num_pixels / 2)
        for _ in range(num_pepper):
            x, y = np.random.randint(0, h), np.random.randint(0, w)
            img_array[x, y] = 0.0

        return torch.tensor(img_array).permute(2, 0, 1)  # Convert back to CHW

    def __call__(self, img):
        # Apply resize
        img = self.resize(img)

        # Randomly select one augmentation
        augmentation_choices = [
            lambda x: self.random_crop(x),
            lambda x: self.random_rotation(x),
            lambda x: self.color_jitter(x),
            lambda x: self.add_gaussian_noise(self.to_tensor(x)),
            lambda x: self.add_salt_and_pepper_noise(self.to_tensor(x))
        ]
        selected_augmentation = random.choice(augmentation_choices)

        # Apply the selected augmentation
        img = selected_augmentation(img)

        # If augmentation returned a tensor (e.g., Gaussian noise), ensure further transformations
        if isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            img_tensor = self.to_tensor(img)

        # Normalize
        img_tensor = self.normalize(img_tensor)
        return img_tensor


# Load full dataset
full_dataset = datasets.ImageFolder(root='C:/Users/shirs/Downloads/CustomTrain', transform=transform)


# Utility function to get class indices
def get_class_indices(dataset, class_name):
    class_idx = dataset.class_to_idx[class_name]
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
    return indices


# Split dataset
real_indices = get_class_indices(full_dataset, 'REAL')
fake_indices = get_class_indices(full_dataset, 'FAKE')

random.seed(42)
real_test_selected = random.sample(real_indices, 3000)
fake_test_selected = random.sample(fake_indices, 3000)

test_selected_indices = real_test_selected + fake_test_selected
test_subset = Subset(full_dataset, test_selected_indices)

test_subset.dataset.transform = AugmentationTransform()

# Create DataLoaders
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

# Print sizes of subsets
print(f"Validation dataset size: {len(test_subset)}")

# Validate with tqdm
print("Processing validation dataset:")
for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Validation Progress", unit="batch")):
    # Here, images and labels are processed as needed (e.g., pass them to the model)
    pass
