"""DINOv2 patch-token descriptor — a no-cross-view-training control.

This model has never seen a UAV-to-satellite pair, or any retrieval objective. It
exists in the benchmark to answer one question: how much of a cross-view model's
score comes from cross-view training, and how much from generic visual features?
It is the control condition, not a recommended encoder.

Descriptors are the mean of the patch tokens, matching how avl.models.denseuav_vit
pools its backbone, so the comparison isolates the weights rather than the pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
import torchvision.transforms as T


BACKBONE = "vit_base_patch14_dinov2.lvd142m"
IMG_SIZE = 224
DINOV2_DIM = 768


class DINOv2Descriptor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=True,
            num_classes=0,
            img_size=IMG_SIZE,
        )
        self.num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(images)
        return tokens[:, self.num_prefix_tokens:].mean(dim=1)


def load_dinov2(device: torch.device) -> DINOv2Descriptor:
    return DINOv2Descriptor().to(device).eval()


_cfg = timm.get_pretrained_cfg(BACKBONE)

DINOV2_TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=_cfg.mean, std=_cfg.std),
    ]
)
