# Imports
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim


# Model Configuration
base_model = [
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
]

phi_values = {
    "b0": (0, 32, 0.2),
    "b1": (1, 32, 0.2),
    "b2": (2, 32, 0.3),
    "b3": (3, 32, 0.3),
    "b4": (4, 32, 0.4),
}


# Model Components
class CNNBlock(nn.Module):
    """
    A convolutional block with BatchNorm and SiLU activation.
    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Kernel size for the convolution.
        stride (int): Stride for the convolution.
        padding (int): Padding for the convolution.
        groups (int): Number of groups for grouped convolutions.
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channel)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    Args:
        in_channel (int): Number of input channels.
        reduced_dim (int): Reduced dimensionality for the hidden layer.
    """
    def __init__(self, in_channel, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block for EfficientNet.
    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Kernel size for depthwise convolution.
        stride (int): Stride for depthwise convolution.
        padding (int): Padding for depthwise convolution.
        expand_ratio (int): Expansion ratio for the hidden dimension.
        reduction (int): Reduction ratio for the Squeeze-and-Excitation block.
        survival_prob (float): Probability for stochastic depth.
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channel == out_channel and stride == 1
        hidden_dim = in_channel * expand_ratio
        reduced_dim = int(in_channel / reduction)
        self.expand = in_channel != hidden_dim

        if self.expand:
            self.expand_conv = CNNBlock(in_channel, hidden_dim, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            CNNBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return x / self.survival_prob * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        return self.stochastic_depth(self.conv(x)) + inputs if self.use_residual else self.conv(x)


class EfficientNet(nn.Module):
    """
    EfficientNet Model with configurable scaling.
    Args:
        version (str): Version of the EfficientNet (e.g., "b0").
    """
    def __init__(self, version):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = math.ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.fc_layers = nn.Sequential(
            nn.Linear(last_channels, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU6(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, _, drop_rate = phi_values[version]
        return beta ** phi, alpha ** phi, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=1, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * math.ceil(int(channels * width_factor) / 4)
            layers_repeats = math.ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )
                )
                in_channels = out_channels

        features.append(CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        x = x.view(x.shape[0], -1)
        return self.fc_layers(x)

