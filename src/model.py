import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import config


class ScSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze-Excitation (scSE).
    Roy et al. 2019 — preserved from original architecture.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Squeeze-Excitation
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )
        # Spatial Squeeze-Excitation
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        cse_w = self.cse(x).view(b, c, 1, 1)
        sse_w = self.sse(x)
        return (cse_w * x) + (sse_w * x)


class OilSpillModel(nn.Module):
    """
    DeepLabV3+ with EfficientNet-B4 encoder + scSE attention.
    Replaces MobileNetV3-Large backbone for higher accuracy.
    EfficientNet-B4: 19M params vs MobileNetV3: 11M params.
    Compound scaling gives better feature extraction at minimal cost.
    """
    def __init__(self):
        super().__init__()

        # EfficientNet-B4 backbone with DeepLabV3+ decoder
        self.base = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=256,          # output features, not final classes
            activation=None,
            decoder_atrous_rates=(6, 12, 18),
        )

        # scSE on top of decoder output
        self.scse = ScSEBlock(channels=256, reduction=16)

        # Final classification head
        self.head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder + decoder
        features = self.base(x)                     # (B, 256, H/4, W/4)

        # scSE attention
        features = self.scse(features)

        # Classification head
        logits = self.head(features)                # (B, 1, H/4, W/4)

        # Upsample to input resolution
        logits = F.interpolate(
            logits, size=input_size,
            mode='bilinear', align_corners=False
        )
        return logits


def get_model(device):
    model = OilSpillModel()
    return model.to(device)


# ── V1 Model (MobileNetV3-Large backbone) — kept for checkpoint comparison ───

class OilSpillModelV1(nn.Module):
    """
    Original DeepLabV3+ with MobileNetV3-Large encoder + scSE attention.
    Used in Run 1-4 (baseline). Preserved here to allow loading old checkpoints
    for side-by-side comparison with OilSpillModel (EfficientNet-B4).

    Architecture:
        MobileNetV3-Large backbone (~11M params, pretrained on ImageNet)
        + scSE attention block
        + DeepLabV3+ ASPP head
    Input:  256×256×3
    Output: 256×256×1 raw logits (apply sigmoid for probability)
    """
    def __init__(self):
        super().__init__()

        self.base = smp.DeepLabV3Plus(
            encoder_name="timm-mobilenetv3_large_100",
            encoder_weights="imagenet",
            in_channels=3,
            classes=256,
            activation=None,
            decoder_atrous_rates=(6, 12, 18),
        )

        self.scse = ScSEBlock(channels=256, reduction=16)

        self.head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.base(x)
        features = self.scse(features)
        logits = self.head(features)
        logits = F.interpolate(
            logits, size=input_size,
            mode='bilinear', align_corners=False
        )
        return logits


def get_model_v1(device):
    """Load MobileNetV3-Large (V1) model. Use with old best_model.pth checkpoints."""
    model = OilSpillModelV1()
    return model.to(device)