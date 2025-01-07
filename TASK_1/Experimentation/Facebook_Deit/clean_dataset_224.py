import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from PIL import Image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.LANCZOS),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root='/kaggle/input/realfake-data/train', transform=transform)


def get_class_indices(dataset, class_name):
    class_idx = dataset.class_to_idx[class_name]
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
    return indices


real_indices = get_class_indices(full_dataset, 'REAL')
fake_indices = get_class_indices(full_dataset, 'FAKE')

random.seed(42)

real_val_selected = random.sample(real_indices, 5000)
fake_val_selected = random.sample(fake_indices, 5000)

val_selected_indices = real_val_selected + fake_val_selected

val_subset = Subset(full_dataset, val_selected_indices)

train_indices = [i for i in range(len(full_dataset)) if i not in val_selected_indices]
train_subset = Subset(full_dataset, train_indices)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

# Print sizes of subsets
print(f"Train dataset size: {len(train_subset)}")
print(f"Validation dataset size: {len(val_subset)}")
