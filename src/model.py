import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models

class ChannelSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialSE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.conv(x))
        return x * y

class scSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cse = ChannelSE(channels, reduction)
        self.sse = SpatialSE(channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)

class OilSpillDeepLab(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        base = seg_models.deeplabv3_mobilenet_v3_large(
            weights="DEFAULT"
        )
        self.backbone = base.backbone
        self.classifier = base.classifier
        self.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.scse = scSEBlock(channels=960)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.backbone(x)
        high = features.get("out", features.get("high", None))
        if high is None:
            # Fallback if dictionary keys are different
            high = list(features.values())[-1]
        high = self.scse(high)
        out = self.classifier(high)
        out = nn.functional.interpolate(
            out, size=input_size, mode='bilinear', align_corners=True
        )
        return out

def get_model(device):
    model = OilSpillDeepLab(num_classes=1)
    return model.to(device)