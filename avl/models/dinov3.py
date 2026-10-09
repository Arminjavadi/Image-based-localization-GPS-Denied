"""DINOv3 + GeM — a frozen foundation-model descriptor, nothing trained or fitted.

Pooling follows VFM-Loc (Ding et al., 2026, "Training-Free Cross-View
Geo-Localization via Aligning Discriminative Visual Hierarchies"): GeM (p=3) over
the final normalised patch tokens, plus a 2x2 R-MAC grid down-weighted by
1/level**alpha. VFM-Loc's Procrustes alignment step is *not* reproduced: it fits
the rotation on ground-truth drone/satellite pairs from the test set, which a
GPS-denied flight does not have.

Two backbones are exposed:

  dinov3-b   ViT-B/16, web images (LVD-1689M)
  dinov3-l   ViT-L/16, web images (LVD-1689M)

anyloc-sat already covers the satellite-pretrained (SAT-493M) ViT-L with VLAD.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as T


BACKBONES = {
    "dinov3-b": ("hf_hub:timm/vit_base_patch16_dinov3.lvd1689m", 768),
    "dinov3-l": ("hf_hub:timm/vit_large_patch16_dinov3.lvd1689m", 1024),
}
IMG_SIZE = 320  # 20x20 patches of 16 px
GEM_P = 3.0
RMAC_LEVELS = (1, 2)
RMAC_ALPHA = 6.0


def _gem(x: torch.Tensor, dims) -> torch.Tensor:
    return x.clamp_min(1e-6).pow(GEM_P).mean(dim=dims).pow(1.0 / GEM_P)


WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "data" / "weights" / "dinov3"


class DINOv3GeM(nn.Module):
    def __init__(self, backbone_name: str) -> None:
        super().__init__()
        # Prefer a local copy (data/weights/dinov3/<timm name>.safetensors): the hub
        # download is large and stalls on slow links.
        # timm always goes to the Hub for an "hf_hub:" name, so use the plain name then.
        timm_name = backbone_name.removeprefix("hf_hub:timm/")
        local = WEIGHTS_DIR / f"{timm_name}.safetensors"
        if local.exists():
            backbone_name, overlay = timm_name, {"file": str(local)}
        else:
            overlay = None
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, img_size=IMG_SIZE,
            pretrained_cfg_overlay=overlay,
        )
        self.num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(images)[:, self.num_prefix_tokens:]
        b, n, c = tokens.shape
        side = int(round(n ** 0.5))
        grid = tokens.transpose(1, 2).reshape(b, c, side, side)
        out = torch.zeros(b, c, device=tokens.device, dtype=tokens.dtype)
        for level in RMAC_LEVELS:
            weight = 1.0 / level ** RMAC_ALPHA
            for rows in torch.tensor_split(torch.arange(side), level):
                for cols in torch.tensor_split(torch.arange(side), level):
                    region = grid[:, :, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
                    out = out + weight * F.normalize(_gem(region, (-2, -1)), dim=1)
        return out


def load_dinov3(name: str, device: torch.device) -> DINOv3GeM:
    return DINOv3GeM(BACKBONES[name][0]).to(device).eval()


# DINOv3 web checkpoints use ImageNet statistics.
DINOV3_TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
