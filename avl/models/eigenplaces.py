"""EigenPlaces descriptor model — CosPlace's successor.

EigenPlaces (Berton et al., ICCV 2023) trains the same large-scale classification
recipe as CosPlace but mines the training classes along the scene's principal
viewpoint axes, so the descriptor is more robust to the viewpoint change between a
query and the map. It is a drop-in upgrade of the `cosplace` backend and is loaded
the same way, from the authors' Torch Hub entry.

Weights: https://github.com/gmberton/EigenPlaces  (get_trained_model)
"""

from __future__ import annotations

import os

import torch
import torchvision.transforms as T


BACKBONE = "ResNet50"
DESCRIPTOR_DIM = 2048

EIGENPLACES_TRANSFORM = T.Compose(
    [
        T.Resize((322, 322), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _trust_hub_repos(*repos: str) -> None:
    """Pre-authorise Torch Hub repos so nested loads don't block on an input() prompt.

    EigenPlaces' hubconf builds its backbone with an un-flagged
    ``torch.hub.load("gmberton/cosplace", ...)``, which raises EOFError in a
    non-interactive process unless that repo is already on the trusted list.
    """
    trusted_list = os.path.join(torch.hub.get_dir(), "trusted_list")
    os.makedirs(torch.hub.get_dir(), exist_ok=True)
    have: set[str] = set()
    if os.path.exists(trusted_list):
        have = {line.strip() for line in open(trusted_list)}
    missing = [r.replace("/", "_") for r in repos if r.replace("/", "_") not in have]
    if missing:
        with open(trusted_list, "a") as handle:
            handle.write("\n".join(missing) + "\n")


def load_eigenplaces(device: torch.device) -> torch.nn.Module:
    _trust_hub_repos("gmberton/cosplace", "gmberton/EigenPlaces")
    model = torch.hub.load(
        "gmberton/EigenPlaces",
        "get_trained_model",
        backbone=BACKBONE,
        fc_output_dim=DESCRIPTOR_DIM,
        trust_repo=True,
    )
    return model.to(device).eval()
