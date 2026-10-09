"""dinov2-ft — frozen DINOv2 ViT-B/14 + a retrieval head trained on UAV-VisLoc.

The backbone is untouched; only the small head (avl.finetune.head) is trained, on
drone-frame / satellite-crop pairs from UAV-VisLoc regions other than the test
regions (scripts/finetune_features.py, scripts/finetune_head.py). The head's
checkpoint path comes from ``AVLConfig.head_weights``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from avl.finetune.backbone import TRANSFORM as DINOV2_FT_TRANSFORM, TokenBackbone
from avl.finetune.head import RetrievalHead

__all__ = ["DINOV2_FT_TRANSFORM", "DINOv2FT", "load_dinov2_ft"]


class DINOv2FT(nn.Module):
    def __init__(self, head_weights: Path) -> None:
        super().__init__()
        if not Path(head_weights).is_file():
            raise FileNotFoundError(
                f"dinov2-ft head weights not found at {head_weights}; train one with "
                "scripts/finetune_head.py or pass --head-weights"
            )
        self.backbone = TokenBackbone()
        self.head = RetrievalHead.load(head_weights)
        self.output_dim = int(self.head.config["out_dim"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))


def load_dinov2_ft(head_weights: Path, device: torch.device) -> DINOv2FT:
    return DINOv2FT(head_weights).to(device).eval()
