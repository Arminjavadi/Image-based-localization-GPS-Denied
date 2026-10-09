"""Game4Loc / GTA-UAV descriptor model.

A UAV-to-satellite cross-view retrieval model (AAAI 2025) built on a RoPE ViT-B/16
backbone. Unlike the DenseUAV checkpoint, which was trained on 14 campuses in one
city, this one is trained on a large synthetic corpus plus real imagery and is
published in a *cross-area* variant whose train and test regions are disjoint —
which makes it the natural comparison when the question is generalisation.

Weights: https://huggingface.co/Yux1ang/gta_uav_pretrained_models
Code:    https://github.com/Yux1angJi/GTA-UAV
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError


REPO_ID = "Yux1ang/gta_uav_pretrained_models"
CHECKPOINT_NAME = "vit_base_eva_gta_cross_area.pth"
BACKBONE = "vit_base_patch16_rope_reg1_gap_256.sbb_in1k"
IMG_SIZE = 384
DESCRIPTOR_DIM = 768


class Game4LocViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = timm.create_model(
            BACKBONE,
            pretrained=False,
            num_classes=0,
            img_size=IMG_SIZE,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def _backbone_state(state_dict: dict) -> dict:
    """Unwrap DesModel (and possibly DDP); drop training-only tensors."""
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(("logit_scale", "offset"))
    }
    for prefix in ("module.", "model1.", "model."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _checkpoint_path() -> str:
    """Prefer a checkpoint vendored under data/weights/, else pull from the Hub."""
    local = Path(__file__).resolve().parents[2] / "data" / "weights" / CHECKPOINT_NAME
    if local.exists():
        return str(local)
    try:
        return hf_hub_download(REPO_ID, CHECKPOINT_NAME, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(REPO_ID, CHECKPOINT_NAME)


def load_game4loc(device: torch.device) -> Game4LocViT:
    state_dict = torch.load(_checkpoint_path(), map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = _backbone_state(state_dict)

    model = Game4LocViT()
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if "rope" not in k]
    if real_missing:
        raise RuntimeError(f"Game4Loc checkpoint is missing backbone weights: {real_missing[:8]}")
    return model.to(device).eval()


_cfg = timm.get_pretrained_cfg(BACKBONE)

GAME4LOC_TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=_cfg.mean, std=_cfg.std),
    ]
)
