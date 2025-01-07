import random
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset


# ----------------------------- Custom Augmentation Class -----------------------------

class CustomAugmentation:
    """
    Implements various custom augmentation techniques for image preprocessing.
    """

    def gaussian_blur(self, image: Image.Image) -> Image.Image:
        """
        Applies Gaussian blur to the image.

        Args:
            image (PIL.Image): Input image to be blurred.

        Returns:
            PIL.Image: Blurred image.
        """
        blur = transforms.GaussianBlur(5, sigma=(0.5, 2.0))
        return blur(image)

    def salt_and_pepper_noise(self, image: Image.Image, salt_prob=0.02, pepper_prob=0.02) -> Image.Image:
        """
        Adds salt-and-pepper noise to the image.

        Args:
            image (PIL.Image): Input image.
            salt_prob (float): Probability of adding salt noise.
            pepper_prob (float): Probability of adding pepper noise.

        Returns:
            PIL.Image: Image with added salt-and-pepper noise.
        """
        np_image = np.array(image)
        total_pixels = np_image.size
        num_salt = int(salt_prob * total_pixels)
        num_pepper = int(pepper_prob * total_pixels)

        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in np_image.shape]
        np_image[salt_coords[0], salt_coords[1], :] = 255

        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in np_image.shape]
        np_image[pepper_coords[0], pepper_coords[1], :] = 0

        return Image.fromarray(np_image)

    def add_gaussian_noise(self, image: Image.Image, mean=0, std=0.05) -> Image.Image:
        """
        Adds Gaussian noise to the image.

        Args:
            image (PIL.Image): Input image.
            mean (float): Mean of Gaussian noise.
            std (float): Standard deviation of Gaussian noise.

        Returns:
            PIL.Image: Image with added Gaussian noise.
        """
        np_image = np.array(image).astype(np.float32)
        noise = np.random.normal(mean, std * 255, np_image.shape)
        np_image = np.clip(np_image + noise, 0, 255)
        return Image.fromarray(np_image.astype(np.uint8))

    def adjust_brightness(self, image: Image.Image) -> Image.Image:
        """
        Adjusts the brightness of the image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            PIL.Image: Brightness-adjusted image.
        """
        brightness = transforms.ColorJitter(brightness=0.2)
        return brightness(image)

    def adjust_contrast(self, image: Image.Image) -> Image.Image:
        """
        Adjusts the contrast of the image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            PIL.Image: Contrast-adjusted image.
        """
        contrast = transforms.ColorJitter(contrast=0.2)
        return contrast(image)

    def affine(self, image: Image.Image) -> Image.Image:
        """
        Applies a random affine transformation to the image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            PIL.Image: Affine-transformed image.
        """
        affine_transform = transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1))
        return affine_transform(image)

    def elastic_transform(self, image: Image.Image, alpha=1, sigma=8) -> Image.Image:
        """
        Applies elastic deformation to the image.

        Args:
            image (PIL.Image): Input image.
            alpha (float): Scaling factor for deformation.
            sigma (float): Standard deviation for Gaussian filter.

        Returns:
            PIL.Image: Elastically transformed image.
        """
        np_image = np.array(image)
        random_state = np.random.RandomState(None)
        shape = np_image.shape
        dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        for c in range(shape[2]):
            np_image[..., c] = map_coordinates(np_image[..., c], indices, order=1, mode='reflect').reshape(shape[:2])

        return Image.fromarray(np_image)


# ----------------------------- Dataset Preparation -----------------------------

# Define base transformations
base_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Resize((224, 224), interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define augmented transformations
custom_augmentation = CustomAugmentation()
augmentation = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Resize((224, 224), interpolation=Image.LANCZOS),
    transforms.RandomChoice([
        custom_augmentation.gaussian_blur,
        custom_augmentation.salt_and_pepper_noise,
        custom_augmentation.add_gaussian_noise,
        custom_augmentation.adjust_brightness,
        custom_augmentation.adjust_contrast,
        custom_augmentation.affine,
        custom_augmentation.elastic_transform
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset and split into augmented and non-augmented subsets
full_dataset = datasets.ImageFolder(root='/kaggle/input/final-datasethopefully', transform=base_transform)
total_size = len(full_dataset)
augment_size = int(0.4 * total_size)
no_augment_size = total_size - augment_size

indices = list(range(total_size))
random.shuffle(indices)
augment_indices = indices[:augment_size]
no_augment_indices = indices[augment_size:]

augment_subset = Subset(datasets.ImageFolder(root='/kaggle/input/final-datasethopefully', transform=augmentation),
                        augment_indices)
no_augment_subset = Subset(full_dataset, no_augment_indices)

combined_dataset = ConcatDataset([no_augment_subset, augment_subset])

# Create DataLoader
train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)


print(f"Train dataset size: {len(combined_dataset)}")
print(f"Augmented data size: {len(augment_subset)}")
