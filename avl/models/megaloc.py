"""MegaLoc descriptor model — one retrieval model for many localization domains.

MegaLoc (Berton & Masone, CVPR 2025 workshops) is a DINOv2-B backbone with SALAD
(optimal-transport) aggregation, trained on a union of street-view, landmark,
indoor and aerial retrieval data. Unlike CosPlace/EigenPlaces it is not
street-view only, which makes it the strongest off-the-shelf general VPR model to
try zero-shot on drone-to-satellite retrieval. Nothing is fitted on the map.

Weights: https://github.com/gmberton/MegaLoc  (get_trained_model)
"""

from __future__ import annotations

import torch
import torchvision.transforms as T

from avl.models.eigenplaces import _trust_hub_repos


DESCRIPTOR_DIM = 8448

# The authors' evaluation preprocessing (README): ImageNet normalisation, 322x322.
MEGALOC_TRANSFORM = T.Compose(
    [
        T.Resize((322, 322), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_megaloc(device: torch.device) -> torch.nn.Module:
    _trust_hub_repos("gmberton/MegaLoc")
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
    return model.to(device).eval()
