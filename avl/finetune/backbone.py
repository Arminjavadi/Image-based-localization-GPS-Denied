"""Frozen DINOv2 ViT-B/14 token features, the input to the trainable head.

A feature is the final-layer CLS token plus the 16 x 16 patch grid average-pooled
to 8 x 8: (65, 768) per image. Keeping a coarse grid rather than one pooled vector
lets the head learn spatial layout (road junctions, field boundaries), which is
most of what distinguishes neighbouring tiles from a drone view; 8 x 8 keeps the
cache small enough to train on CPU from RAM.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

BACKBONE = "vit_base_patch14_dinov2.lvd142m"
IMG_SIZE = 224
EMBED_DIM = 768
GRID = 8
N_TOKENS = 1 + GRID * GRID

_cfg = timm.get_pretrained_cfg(BACKBONE)
TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=_cfg.mean, std=_cfg.std),
    ]
)


class TokenBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vit = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=IMG_SIZE)
        self.prefix = int(getattr(self.vit, "num_prefix_tokens", 1))

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features(images)
        cls = tokens[:, :1]
        side = IMG_SIZE // 14
        patches = tokens[:, self.prefix :].reshape(-1, side, side, EMBED_DIM).permute(0, 3, 1, 2)
        grid = nn.functional.adaptive_avg_pool2d(patches, GRID).flatten(2).transpose(1, 2)
        return torch.cat([cls, grid], dim=1)  # (B, 65, 768)


def load_backbone(device: torch.device | str = "cpu") -> TokenBackbone:
    return TokenBackbone().to(device).eval()


def square_crop(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def open_query(path: str, min_side: int = 448) -> Image.Image:
    """Drone frames are ~4000 px; decode JPEGs at reduced scale (the model sees 224)."""
    image = Image.open(path)
    image.draft("RGB", (min_side * 2, min_side * 2))
    return square_crop(image.convert("RGB"))


def query_rotations(image: Image.Image) -> list[Image.Image]:
    """The same four orientations the retrieval pipeline searches."""
    return [image.rotate(90 * r, expand=True) for r in range(4)]
