import os
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================ Model Architecture ============================

class ConvBn(nn.Module):
    """
    Convolutional layer followed by Batch Normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass through convolution and batch normalization layers.

        Args:
            inp (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Output tensor after convolution and batch normalization.
        """
        return self.batch_norm(self.conv(inp))


class Type1(nn.Module):
    """
    A block consisting of a convolutional layer followed by ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convbn = ConvBn(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass through the Type1 block.

        Args:
            inp (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after convolution, batch norm, and ReLU.
        """
        return self.relu(self.convbn(inp))


class Type2(nn.Module):
    """
    A block that applies a skip connection and a convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(in_channels, out_channels)

    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass through the Type2 block.

        Args:
            inp (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying skip connection and convolution.
        """
        return inp + self.convbn(self.type1(inp))


class Type3(nn.Module):
    """
    A block consisting of a 1x1 convolution, batch norm, and pooling layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(out_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass through the Type3 block.

        Args:
            inp (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after convolution, batch norm, and pooling.
        """
        out = self.batch_norm(self.conv1(inp))
        out1 = self.pool(self.convbn(self.type1(inp)))
        return out + out1


class Type4(nn.Module):
    """
    A block consisting of a global average pooling layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(out_channels, out_channels)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass through the Type4 block.

        Args:
            inp (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after global average pooling.
        """
        return self.gap(self.convbn(self.type1(inp)))


class Srnet(nn.Module):
    """
    SRNet model for image classification.

    Args:
        None
    """
    def __init__(self) -> None:
        super().__init__()

        # Initial layers
        self.type1s = nn.Sequential(Type1(3, 32), Type1(32, 16))

        # Middle layers
        self.type2s = nn.Sequential(
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
        )
        self.type3s = nn.Sequential(
            Type3(16, 16),
            Type3(16, 64),
            Type3(64, 128),
            Type3(128, 256),
        )
        self.type4 = Type4(256, 512)

        # Fully connected layers
        self.dense = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 31),
            nn.ReLU(),
            nn.Linear(31, 2)
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass through the entire SRNet model.

        Args:
            inp (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the network and applying softmax.
        """
        out = self.type1s(inp)
        out = self.type2s(out)
        out = self.type3s(out)
        out = self.type4(out)
        out = out.view(out.size(0), -1)  # Flatten before passing to dense layers
        out = self.dense(out)
        return self.softmax(out)


