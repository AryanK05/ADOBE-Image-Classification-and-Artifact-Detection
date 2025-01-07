import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from PIL import Image
import io
from tqdm import tqdm


# Define transformations for training dataset
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define transformations for validation dataset with JPEG compression
class JpegCompressionTransform:
    def __init__(self, quality=66):
        self.quality = quality
        self.resize = transforms.Resize((224, 224), interpolation=Image.LANCZOS)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img):
        # Resize image
        img = self.resize(img)

        # Apply JPEG compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        img = Image.open(buffer)

        # Convert to tensor and normalize
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img


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


# Update validation transform for JPEG compression
test_subset.dataset.transform = JpegCompressionTransform(quality=66)

test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

print(f"Validation dataset size: {len(test_subset)}")

# Validate with tqdm
print("Processing validation dataset:")
for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Validation Progress", unit="batch")):
    # Here, images and labels are processed as needed (e.g., pass them to the model)
    pass
