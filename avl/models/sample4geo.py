"""Sample4Geo — ConvNeXt cross-view UAV↔satellite retrieval.

Sample4Geo (Deuser, Habel, Oswald; ICCV 2023) trains a plain ConvNeXt-Base with a
symmetric InfoNCE loss and hard-negative sampling on cross-view pairs. It holds up
well across datasets, which makes it the natural strong baseline next to
`game4loc` when the question is generalisation to a new region.

The eval checkpoint is a `state_dict` of the repo's ``TimmModel`` wrapper:
``timm.create_model("convnext_base.fb_in22k_ft_in1k_384", num_classes=0)`` plus a
training-only ``logit_scale`` scalar. With ``num_classes=0`` the backbone returns
its 1024-d globally pooled feature, which is the descriptor.

Weights: the "University-1652" checkpoint (``weights_e1_0.9515.pth``) from the
Google-Drive folder linked in https://github.com/Skyy93/Sample4Geo
Provide it as ``data/weights/sample4geo_university1652_convnext_base.pth`` (or let
this module fetch the folder with gdown on first use).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T


BACKBONE = "convnext_base.fb_in22k_ft_in1k_384"
IMG_SIZE = 384
DESCRIPTOR_DIM = 1024

WEIGHTS_NAME = "sample4geo_university1652_convnext_base.pth"
GDRIVE_FOLDER_ID = "1PMuUqvDnCb216D8_ZDDJzDD3FxeH5BoA"
GDRIVE_FILE_ID = ""  # set to the direct file id if it becomes known — skips the folder pull

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE = Path.home() / ".cache" / "avl" / WEIGHTS_NAME


class Sample4GeoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = timm.create_model(BACKBONE, pretrained=False, num_classes=0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def _backbone_state(state_dict: dict) -> dict:
    """Unwrap the repo's TimmModel: drop logit_scale, strip a leading module./model. prefix."""
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("logit_scale")}
    for prefix in ("module.", "model."):
        if state_dict and all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _fetch_from_drive() -> Path | None:
    try:
        import gdown
    except ImportError:
        return None
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    if GDRIVE_FILE_ID:
        gdown.download(id=GDRIVE_FILE_ID, output=str(_CACHE), quiet=False)
        return _CACHE if _CACHE.exists() else None
    # No direct file id: pull the whole weights folder, then keep the U1652 ConvNeXt file.
    staging = _CACHE.parent / "sample4geo_drive"
    try:
        gdown.download_folder(id=GDRIVE_FOLDER_ID, output=str(staging), quiet=False, remaining_ok=True)
    except Exception:  # gdown folder API is flaky; fall through to the manual instructions
        return None
    hits = sorted(staging.rglob("*.pth"))
    pick = next(
        (p for p in hits if "university" in str(p).lower() and "convnext" in str(p).lower()),
        next((p for p in hits if "convnext" in str(p).lower()), None),
    )
    if pick is None:
        return None
    shutil.copy2(pick, _CACHE)
    return _CACHE


def _checkpoint_path() -> str:
    for candidate in (
        _REPO_ROOT / "data" / "weights" / WEIGHTS_NAME,
        _REPO_ROOT / "pretrained" / "university" / BACKBONE / "weights_e1_0.9515.pth",
        _CACHE,
    ):
        if candidate.exists():
            return str(candidate)
    fetched = _fetch_from_drive()
    if fetched is not None and fetched.exists():
        return str(fetched)
    raise FileNotFoundError(
        "Sample4Geo weights not found. Download the University-1652 checkpoint "
        "'weights_e1_0.9515.pth' from the folder linked at "
        "https://github.com/Skyy93/Sample4Geo and save it as:\n  "
        f"{_REPO_ROOT / 'data' / 'weights' / WEIGHTS_NAME}\n"
        "or:  gdown --folder "
        f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID} -O data/weights/sample4geo"
    )


def load_sample4geo(device: torch.device) -> Sample4GeoModel:
    state_dict = torch.load(_checkpoint_path(), map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = _backbone_state(state_dict)

    model = Sample4GeoModel()
    missing, _unexpected = model.model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if not k.startswith(("head.fc", "head.flatten"))]
    if real_missing:
        raise RuntimeError(
            f"Sample4Geo checkpoint is missing backbone weights: {real_missing[:8]}"
        )
    return model.to(device).eval()


_cfg = timm.get_pretrained_cfg(BACKBONE)

SAMPLE4GEO_TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=_cfg.mean, std=_cfg.std),
    ]
)
