"""InfoGeo — cross-view *generalizable* UAV geo-localization (Zhang et al., ICML 2026).

Trained on one UAV dataset and evaluated on others, which is the situation here: a
new region the model has never seen. Architecture at inference:

  DINOv2 ViT-B/14, fine-tuned, with one extra learnt "view token" (the authors'
  DINOv2 fork), no final LayerNorm, 448x448 input -> 32x32 patch grid
  -> MixVPR-style head (2 token mixers over the 1024 positions, 768->1024 channel
     projection, 1024->4 row projection) -> 4096-d descriptor.

The slot-attention modules in the checkpoint (slot_model, slot_moe, mix_fusion) are
training-time only: the released inference path has them commented out ("for
distillation only"). The released repository also omits the backbone helper and the
DINOv2 fork, so the backbone is rebuilt here from the official DINOv2 code plus the
two extra tensors in the checkpoint (view_token, pos_view_embed). Token order does
not matter to attention, so only which outputs are read back does.

Weights (Google Drive, linked from https://github.com/HRT00/Official_InfoGeo):
  data/weights/infogeo/infogeo_gta.pth       trained on GTA-UAV   -> model "infogeo"
  data/weights/infogeo/infogeo_denseuav.pth  trained on DenseUAV  -> model "infogeo-dense"
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


DESCRIPTOR_DIM = 4096
IMG_SIZE = 448
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "data" / "weights" / "infogeo"
CHECKPOINTS = {
    "infogeo": "infogeo_gta.pth",
    "infogeo-dense": "infogeo_denseuav.pth",
}


class _MixerLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(x)


class _SlotMixVPR(nn.Module):
    """The checkpoint's `slot_mixvpr` head (MixVPR over the flattened patch grid)."""

    def __init__(self, in_channels: int = 768, hw: int = 1024, out_channels: int = 1024, out_rows: int = 4) -> None:
        super().__init__()
        self.mix = nn.Sequential(_MixerLayer(hw), _MixerLayer(hw))
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B, C, H*W
        x = self.mix(x)
        x = self.channel_proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.row_proj(x)
        return F.normalize(x.flatten(1), p=2, dim=-1)


class InfoGeo(nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        vit = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=False, trust_repo=True
        )
        vit.norm = nn.Identity()  # removed in the authors' backbone
        vit.head = nn.Identity()
        vit.view_token = nn.Parameter(torch.zeros(1, 1, vit.embed_dim))
        vit.pos_view_embed = nn.Parameter(torch.zeros(1, 1, vit.embed_dim))
        self.vit = vit
        self.head = _SlotMixVPR(in_channels=vit.embed_dim)

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        prefix = "model.backbone.dino_model."
        vit_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        head_state = {k[len("slot_mixvpr."):]: v for k, v in state.items() if k.startswith("slot_mixvpr.")}
        missing, unexpected = self.vit.load_state_dict(vit_state, strict=False)
        missing = [k for k in missing if not k.startswith(("norm.", "head."))]
        if missing or unexpected:
            raise RuntimeError(f"InfoGeo backbone mismatch: missing={missing} unexpected={unexpected}")
        self.head.load_state_dict(head_state, strict=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        b = images.shape[0]
        x = vit.prepare_tokens_with_masks(images)  # [cls, patches] + interpolated pos-embed
        view = (vit.view_token + vit.pos_view_embed).expand(b, -1, -1)
        x = torch.cat((x[:, :1], view, x[:, 1:]), dim=1)
        for block in vit.blocks:
            x = block(x)
        patches = x[:, 2:]  # drop cls + view token; B, H*W, C
        return self.head(patches.transpose(1, 2))


def load_infogeo(name: str, device: torch.device) -> InfoGeo:
    checkpoint = WEIGHTS_DIR / CHECKPOINTS[name]
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"InfoGeo weights not found at {checkpoint}. Download them from the Google Drive "
            "folder linked in https://github.com/HRT00/Official_InfoGeo (README, 'Model Checkpoints')."
        )
    return InfoGeo(checkpoint).to(device).eval()


# The authors' eval transform: resize to 448x448, ImageNet statistics.
INFOGEO_TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
