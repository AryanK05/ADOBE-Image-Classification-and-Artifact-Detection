import random
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm
import torchattacks

# -----------------------------------------------------------
# Section 1: Data Transformations and Dataset Preparation
# -----------------------------------------------------------

# Define data transformations for normalization and resizing
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(
    root='C:/Users/shirs/Downloads/CustomTrain', transform=transform
)

def get_class_indices(dataset, class_name):
    """
    Get indices of samples belonging to a specific class.

    Args:
        dataset (Dataset): The dataset containing images and labels.
        class_name (str): The class name to filter indices for.

    Returns:
        List[int]: Indices of samples belonging to the specified class.
    """
    class_idx = dataset.class_to_idx[class_name]
    return [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]

# Prepare balanced subsets for testing
real_indices = get_class_indices(full_dataset, 'REAL')
fake_indices = get_class_indices(full_dataset, 'FAKE')

random.seed(42)
test_selected_indices = random.sample(real_indices, 3000) + random.sample(fake_indices, 3000)
test_subset = Subset(full_dataset, test_selected_indices)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

# -----------------------------------------------------------
# Section 2: Model and Attack Setup
# -----------------------------------------------------------

# Load pretrained EfficientNet B0 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b0(pretrained=True).to(device)
model.eval()

# Define adversarial attacks using torchattacks
fgsm = torchattacks.FGSM(model, eps=0.1)
pgd = torchattacks.PGD(model, eps=0.1, alpha=0.01, steps=40)

# Define reverse normalization transformation for visualization
reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
])

# -----------------------------------------------------------
# Section 3: Utility Functions for Perturbations
# -----------------------------------------------------------

def apply_jpeg_compression(image, quality=75):
    """
    Apply JPEG compression to an image tensor.

    Args:
        image (Tensor): Input image tensor.
        quality (int): JPEG compression quality (1-100).

    Returns:
        Tensor: Compressed and normalized image tensor.
    """
    image = reverse_transform(image)
    np_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    pil_image = Image.fromarray(np_image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return transform(compressed_image)

def perturb_dataset(loader, fgsm, pgd, jpeg_quality=75, fgsm_ratio=0.0725, pgd_ratio=0.075, jpeg_ratio=0.125):
    """
    Generate a perturbed dataset by applying FGSM, PGD, or JPEG compression.

    Args:
        loader (DataLoader): DataLoader for the dataset to perturb.
        fgsm (torchattacks.Attack): FGSM attack instance.
        pgd (torchattacks.Attack): PGD attack instance.
        jpeg_quality (int): JPEG compression quality.
        fgsm_ratio (float): Proportion of data to perturb with FGSM.
        pgd_ratio (float): Proportion of data to perturb with PGD.
        jpeg_ratio (float): Proportion of data to perturb with JPEG compression.

    Returns:
        DataLoader: DataLoader for the perturbed dataset.
    """
    perturbed_inputs = []
    perturbed_labels = []

    total_images = len(loader.dataset)
    fgsm_count = int(total_images * fgsm_ratio)
    pgd_count = int(total_images * pgd_ratio)
    jpeg_count = int(total_images * jpeg_ratio)

    indices = list(range(total_images))
    random.shuffle(indices)

    fgsm_indices = set(indices[:fgsm_count])
    pgd_indices = set(indices[fgsm_count:fgsm_count + pgd_count])
    jpeg_indices = set(indices[fgsm_count + pgd_count:fgsm_count + pgd_count + jpeg_count])

    for i, (inputs, labels) in enumerate(tqdm(loader, desc="Perturbing Dataset")):
        inputs, labels = inputs.to(device), labels.to(device)

        for j in range(inputs.size(0)):
            global_idx = i * loader.batch_size + j

            if global_idx in jpeg_indices:
                perturbed_input = apply_jpeg_compression(inputs[j]).unsqueeze(0)
            elif global_idx in fgsm_indices:
                perturbed_input = fgsm(inputs[j:j + 1], labels[j:j + 1])
            elif global_idx in pgd_indices:
                perturbed_input = pgd(inputs[j:j + 1], labels[j:j + 1])
            else:
                perturbed_input = inputs[j:j + 1]

            perturbed_inputs.append(perturbed_input.cpu())
            perturbed_labels.append(labels[j:j + 1].cpu())

    adv_inputs_tensor = torch.cat(perturbed_inputs, dim=0)
    adv_labels_tensor = torch.cat(perturbed_labels, dim=0)

    perturbed_dataset = TensorDataset(adv_inputs_tensor, adv_labels_tensor)
    return DataLoader(perturbed_dataset, batch_size=64, shuffle=True)

# -----------------------------------------------------------
# Section 4: Perturb and Load Dataset
# -----------------------------------------------------------

perturbed_loader = perturb_dataset(test_loader, fgsm, pgd, jpeg_quality=66)
