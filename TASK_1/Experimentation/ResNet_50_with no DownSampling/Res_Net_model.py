import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# Constants
LIMIT_SIZE = 1536
LIMIT_SLIDE = 1024

class ChannelLinear(nn.Linear):
    """
    A custom linear layer that applies a fully connected transformation on the channels of input tensors.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(ChannelLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.shape[0], 1, 1, -1)
        x = x.flatten(1)  # Flatten the tensor from dimension 1 onwards
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None, :]
        return x.view(x.size(0), -1)


class ResNet50Custom(nn.Module):
    """
    Custom ResNet50 model modified to include ChannelLinear for the final layer.
    """
    def __init__(self, num_classes=1):
        super(ResNet50Custom, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = ChannelLinear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)

    def apply(self, pil, device):
        if pil.size[0] > LIMIT_SIZE and pil.size[1] > LIMIT_SIZE:
            with torch.no_grad():
                img = self.transform(pil).to(device)
                logits, weights = [], []
                for idx0 in range(0, img.shape[-2], LIMIT_SLIDE):
                    for idx1 in range(0, img.shape[-1], LIMIT_SLIDE):
                        clip = img[..., idx0:min(idx0 + LIMIT_SLIDE, img.shape[-2]),
                                   idx1:min(idx1 + LIMIT_SLIDE, img.shape[-1])]
                        logit = torch.squeeze(self(clip[None, :, :, :])).cpu().numpy()
                        weight = clip.shape[-2] * clip.shape[-1]
                        logits.append(logit)
                        weights.append(weight)
                logit = np.mean(np.asarray(logits) * np.asarray(weights)) / np.mean(weights)
        else:
            with torch.no_grad():
                logit = torch.squeeze(self(self.transform(pil).to(device)[None, :, :, :])).cpu().numpy()
        return logit


def resnet50_custom_model(device, num_classes=1):
    """
    Creates an instance of the custom ResNet50 model.
    """
    model = ResNet50Custom(num_classes=num_classes)
    return model.to(device).eval()
