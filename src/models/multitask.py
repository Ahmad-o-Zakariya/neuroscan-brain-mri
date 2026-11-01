"""Architecture used by the ConvNeXt-Tiny and Swin-Tiny multitask checkpoints."""

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskModel(nn.Module):
    """Shared hierarchical encoder with classification and binary-mask heads."""

    STAGE_IDS = (1, 3, 5, 7)

    def __init__(self, architecture: str, input_size: int = 224):
        super().__init__()
        self.architecture = architecture
        self.input_size = int(input_size)
        from torchvision.models import convnext_tiny, swin_t

        if architecture == "convnext_tiny":
            backbone = convnext_tiny(weights=None)
        elif architecture == "swin_tiny":
            backbone = swin_t(weights=None)
        else:
            raise ValueError(f"Unsupported multitask architecture: {architecture}")

        self.encoder = backbone.features
        self.classifier = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 4))
        self.proj = nn.ModuleList(
            [nn.Conv2d(96, 96, 1), nn.Conv2d(192, 128, 1), nn.Conv2d(384, 128, 1), nn.Conv2d(768, 128, 1)]
        )
        self.dec3 = ConvBlock(256, 128)
        self.dec2 = ConvBlock(256, 96)
        self.dec1 = ConvBlock(192, 64)
        self.mask_head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        output_size = x.shape[-2:]
        skips = []
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in self.STAGE_IDS:
                feature = x.permute(0, 3, 1, 2).contiguous() if self.architecture == "swin_tiny" else x
                skips.append(feature)

        f1, f2, f3, f4 = skips
        classification = self.classifier(F.adaptive_avg_pool2d(f4, 1).flatten(1))
        decoder = F.interpolate(self.proj[3](f4), size=f3.shape[-2:], mode="bilinear", align_corners=False)
        decoder = self.dec3(torch.cat([decoder, self.proj[2](f3)], dim=1))
        decoder = F.interpolate(decoder, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        decoder = self.dec2(torch.cat([decoder, self.proj[1](f2)], dim=1))
        decoder = F.interpolate(decoder, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        decoder = self.dec1(torch.cat([decoder, self.proj[0](f1)], dim=1))
        segmentation = F.interpolate(
            self.mask_head(decoder), size=output_size, mode="bilinear", align_corners=False
        )
        return {"classification": classification, "segmentation": segmentation}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
