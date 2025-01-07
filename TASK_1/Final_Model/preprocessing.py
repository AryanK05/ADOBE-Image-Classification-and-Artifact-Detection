import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, ClassLabel
from datasets import Image as img_data
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

def prepare_image_dataset(path, test_size ):
    
    file_names = []
    labels = []

    # Iterate through all image files in the specified directory
    for file in sorted((Path(path).glob('*/*.*'))):
        label = str(file).split('/')[-2]  # Extract the label from the file path
        labels.append(label)  # Add the label to the list
        file_names.append(str(file))  # Add the file path to the list

    # Create a pandas dataframe from the collected file names and labels
    df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
  
    # Convert the dataframe to a HuggingFace Dataset and cast the "image" column to Image
    dataset = Dataset.from_pandas(shuffled_df).cast_column("image", img_data())

    # Initialize ClassLabel with the provided labels
    labels_list=['REAL', 'FAKE']
    ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

    # Mapping labels to IDs
    def map_label2id(example):
        example['label'] = ClassLabels.str2int(example['label'])
        return example

    # Apply the label mapping to the dataset
    dataset = dataset.map(map_label2id, batched=True)

    # Casting label column to ClassLabel Object
    dataset = dataset.cast_column('label', ClassLabels)

    # Splitting the dataset into training and testing sets using the given test_size
    dataset_split = dataset.train_test_split(test_size=test_size, shuffle=True, stratify_by_column="label")

    # Extracting the training and testing data from the split dataset
    train_data = dataset_split['train']
    test_data = dataset_split['test']

    trans_train, trans_test = apply_transformations(train_data, test_data)

    return trans_train, trans_test

from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize, ColorJitter, GaussianBlur, RandomRotation

from PIL import Image

def apply_transformations(train_data, test_data):
    
    input_size = (224, 224)
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5] # We can change the parameters as per our requirement

    # Helper functions for augmentations
    def salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        """
        Adds salt and pepper noise to the image.
        """
        img = np.array(image)
        total_pixels = img.size

        num_salt = int(total_pixels * salt_prob)
        salt_coords = [
            (random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1))
            for _ in range(num_salt)
        ]
        for coord in salt_coords:
            img[coord] = 255  # Setting salt (white pixel)

        
        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [
            (random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1))
            for _ in range(num_pepper)
        ]
        for coord in pepper_coords:
            img[coord] = 0  # Setting pepper (black pixel)

        return Image.fromarray(img)

    def apply_random_augmentations(image):
        
        if random.random() < 0.25:  # 25% probability for Gaussian Blur
            sigma = abs(np.random.normal(0, 3))
            image = GaussianBlur(kernel_size=3, sigma=sigma)(image)

        if random.random() < 0.2:  # 20% probability for Color Jitter
            image = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(image)

        if random.random() < 0.2:  # 20% probability for Salt-and-Pepper Noise
            image = salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01)

        return image

    def apply_random_rotation(image):

        if random.random() < 0.7:  # 70% probability
            degrees = random.uniform(30, 90)
            return RandomRotation(degrees)(image)
        return image

    # Define transformations
    _train_transforms = Compose([
        RandomHorizontalFlip(),
        lambda x: apply_random_rotation(x),  # Apply rotation
        lambda x: apply_random_augmentations(x.convert("RGB")),  # Apply random augmentations
        Resize(input_size),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std),
    ])

    _test_transforms = Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std),
    ])

    # Wrapping transformations for datasets
    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image) for image in examples['image']]
        return examples

    def test_transforms(examples):
        examples['pixel_values'] = [_test_transforms(image) for image in examples['image']]
        return examples

    # Apply transformations
    train_data.set_transform(train_transforms)
    test_data.set_transform(test_transforms)

    return train_data, test_data
