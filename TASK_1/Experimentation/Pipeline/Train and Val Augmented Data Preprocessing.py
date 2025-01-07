import random
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ------------------------------------------------------------
# Data Augmentation and Preprocessing Utilities
# ------------------------------------------------------------

class CustomAugmentation:
    """
    Custom data augmentation class implementing various image transformations.
    """
    def __init__(self):
        self.gaussian_blur = transforms.GaussianBlur(5, sigma=(0.1, 2.0))
        self.motion_blur = transforms.RandomRotation(30)  # Approximation of motion blur
        self.salt_pepper = self.salt_and_pepper_noise
        self.gaussian_noise = self.add_gaussian_noise
        self.poisson_noise = self.add_poisson_noise
        self.adjust_brightness = transforms.ColorJitter(brightness=0.2)
        self.adjust_contrast = transforms.ColorJitter(contrast=0.2)
        self.adjust_saturation = transforms.ColorJitter(saturation=0.2)
        self.adjust_hue = transforms.ColorJitter(hue=0.1)
        self.affine = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2))
        self.elastic_transform = self.elastic_transform_augmentation
        self.perspective_transform = transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
        self.channel_swap = self.channel_swap

    @staticmethod
    def salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        """
        Adds salt and pepper noise to an image.

        Args:
            image (PIL.Image): Input image.
            salt_prob (float): Probability of salt noise.
            pepper_prob (float): Probability of pepper noise.

        Returns:
            PIL.Image: Image with salt and pepper noise applied.
        """
        np_image = np.array(image)
        total_pixels = np_image.size
        num_salt = int(salt_prob * total_pixels)
        num_pepper = int(pepper_prob * total_pixels)

        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in np_image.shape]
        np_image[salt_coords[0], salt_coords[1], :] = 1

        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in np_image.shape]
        np_image[pepper_coords[0], pepper_coords[1], :] = 0

        return Image.fromarray(np_image)

    @staticmethod
    def add_gaussian_noise(image, mean=0, std=0.1):
        """
        Adds Gaussian noise to an image.

        Args:
            image (PIL.Image): Input image.
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.

        Returns:
            PIL.Image: Image with Gaussian noise applied.
        """
        np_image = np.array(image)
        noise = np.random.normal(mean, std, np_image.shape)
        np_image = np.clip(np_image + noise, 0, 255)
        return Image.fromarray(np_image.astype(np.uint8))

    @staticmethod
    def add_poisson_noise(image):
        """
        Adds Poisson noise to an image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            PIL.Image: Image with Poisson noise applied.
        """
        np_image = np.array(image) / 255.0
        vals = np.random.poisson(np_image * 255.0)
        np_image = np.clip(vals, 0, 255).astype(np.uint8)
        return Image.fromarray(np_image)

    @staticmethod
    def elastic_transform_augmentation(image, alpha=1, sigma=10):
        """
        Applies elastic transformation to an image.

        Args:
            image (PIL.Image): Input image.
            alpha (float): Scaling factor for distortion.
            sigma (float): Standard deviation of Gaussian filter.

        Returns:
            PIL.Image: Image with elastic transformation applied.
        """
        np_image = np.array(image)
        random_state = np.random.RandomState(None)
        shape = np_image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0)
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0)
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        distorted_image = map_coordinates(np_image, [y + alpha * dy, x + alpha * dx, z + alpha * dz], order=1, mode='reflect')

        return Image.fromarray(distorted_image)

    @staticmethod
    def channel_swap(image):
        """
        Swaps the Red and Blue channels of an image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            PIL.Image: Image with channels swapped.
        """
        np_image = np.array(image)
        np_image = np_image[..., [2, 1, 0]]
        return Image.fromarray(np_image)


# ------------------------------------------------------------
# Dataset Preparation
# ------------------------------------------------------------

def get_class_indices(dataset, class_name):
    """
    Retrieves indices for a specific class in the dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset object.
        class_name (str): Name of the class.

    Returns:
        List[int]: Indices of samples belonging to the specified class.
    """
    class_idx = dataset.class_to_idx[class_name]
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
    return indices

# Base transformations (without augmentation)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
full_dataset = datasets.ImageFolder(root='/kaggle/input/realfake-data/train', transform=transform)

# Retrieve class indices
real_indices = get_class_indices(full_dataset, 'REAL')
fake_indices = get_class_indices(full_dataset, 'FAKE')

# Select a subset for augmentation
random.seed(42)
real_train_selected = random.sample(real_indices, 11250)
fake_train_selected = random.sample(fake_indices, 11250)

augmented_train_indices = real_train_selected + fake_train_selected
remaining_train_indices = [i for i in range(len(full_dataset)) if i not in augmented_train_indices]

val_indices = random.sample(augmented_train_indices + remaining_train_indices, 10000)
train_indices = [i for i in augmented_train_indices + remaining_train_indices if i not in val_indices]

# Create subsets
train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

# DataLoaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

# Log dataset sizes
print(f"Train dataset size: {len(train_subset)}")
print(f"Validation dataset size: {len(val_subset)}")
