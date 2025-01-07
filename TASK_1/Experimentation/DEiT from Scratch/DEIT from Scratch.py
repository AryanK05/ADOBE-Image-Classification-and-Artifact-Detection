# Imports
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from clean_dataset_224 import train_loader,val_loader
from Test_Dataset_pipeline_224 import test_loader

# ============================ Model Architecture ============================

class HardDistillationLoss(nn.Module):
    """
    Hard distillation loss combining classification loss from the student model
    and teacher supervision through distillation.

    Args:
        teacher (nn.Module): Pre-trained teacher model used for supervision.

    Returns:
        Tensor: Weighted loss combining classification and distillation losses.
    """
    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: Tensor, outputs: tuple[Tensor, Tensor], labels: Tensor) -> Tensor:
        outputs_cls, outputs_dist = outputs
        base_loss = self.criterion(outputs_cls, labels)

        with torch.no_grad():
            teacher_outputs = self.teacher(inputs).logits
        teacher_labels = torch.argmax(teacher_outputs, dim=1)
        teacher_loss = self.criterion(outputs_dist, teacher_labels)

        return 0.5 * base_loss + 0.5 * teacher_loss


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer to split an image into patches and project them into
    embedding space with added class and distillation tokens.

    Args:
        in_channels (int): Number of input image channels (default: 3).
        patch_size (int): Size of each patch (default: 16).
        emb_size (int): Embedding dimension for patches (default: 768).
        img_size (int): Input image size (default: 224).

    Returns:
        Tensor: Patch embeddings with positional encoding.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.dist_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 2, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        dist_tokens = repeat(self.dist_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        x += self.positions
        return x


class ClassificationHead(nn.Module):
    """
    Classification head for student model, providing outputs for both
    class and distillation tokens.

    Args:
        emb_size (int): Embedding size of the input features (default: 768).
        n_classes (int): Number of output classes (default: 1000).

    Returns:
        Tensor: Output logits for class and distillation tokens during training,
                averaged logits during inference.
    """
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__()
        self.head = nn.Linear(emb_size, n_classes)
        self.dist_head = nn.Linear(emb_size, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x, x_dist = x[:, 0], x[:, 1]
        x_head = self.head(x)
        x_dist_head = self.dist_head(x_dist)

        if self.training:
            return x_head, x_dist_head
        return (x_head + x_dist_head) / 2


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism for Transformer Encoder.

    Args:
        emb_size (int): Embedding size of the input features (default: 768).
        num_heads (int): Number of attention heads (default: 8).
        dropout (float): Dropout rate for attention weights (default: 0).

    Returns:
        Tensor: Output of the attention layer.
    """
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class TransformerEncoderBlock(nn.Sequential):
    """
    A single Transformer encoder block combining multi-head attention and
    feedforward layers.

    Args:
        emb_size (int): Embedding size (default: 768).
        drop_p (float): Dropout rate (default: 0).
        forward_expansion (int): Expansion factor for feedforward block (default: 4).
        forward_drop_p (float): Dropout rate for feedforward block (default: 0).
    """
    def __init__(self, emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4, forward_drop_p: float = 0.):
        super().__init__(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size),
                nn.Dropout(drop_p)
            ),
            nn.Sequential(
                nn.LayerNorm(emb_size),
                nn.Linear(emb_size, forward_expansion * emb_size),
                nn.GELU(),
                nn.Dropout(forward_drop_p),
                nn.Linear(forward_expansion * emb_size, emb_size),
                nn.Dropout(forward_drop_p)
            )
        )


class DeiT(nn.Sequential):
    """
    Distilled Vision Transformer (DeiT) model combining patch embedding,
    Transformer encoder, and classification heads.

    Args:
        in_channels (int): Number of input channels (default: 3).
        patch_size (int): Patch size (default: 16).
        emb_size (int): Embedding size (default: 768).
        img_size (int): Input image size (default: 224).
        depth (int): Number of Transformer encoder layers (default: 12).
        n_classes (int): Number of output classes (default: 1000).
    """
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, n_classes=1000):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            nn.Sequential(*[TransformerEncoderBlock(emb_size=emb_size) for _ in range(depth)]),
            ClassificationHead(emb_size, n_classes)
        )

# ============================ Training Code ============================

# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Teacher and student models
processor = ViTImageProcessor.from_pretrained('google/vit-huge-patch14-224-in21k')
teacher = ViTForImageClassification.from_pretrained('google/vit-huge-patch14-224-in21k')
teacher.eval()

student = DeiT(in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, n_classes=10)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher = teacher.to(device)
student = student.to(device)

optimizer = Adam(student.parameters(), lr=1e-4)
criterion = HardDistillationLoss(teacher)

# Training loop
for epoch in range(2):
    student.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = student(inputs)

        # Compute loss
        loss = criterion(inputs, outputs, labels)
        running_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm progress bar
        progress_bar.set_postfix({"Loss": loss.item()})

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")