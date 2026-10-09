"""BoQ — Bag-of-Queries aggregation on a DINOv2-B backbone (Ali-bey et al., CVPR 2024).

A set of learnable global queries cross-attends to the backbone's patch tokens;
the outputs are flattened into the descriptor. Trained on GSV-Cities (street
level), so like MegaLoc it is a general VPR model tried zero-shot on
drone-to-satellite retrieval. Nothing is fitted on the map.

Weights: https://github.com/amaralibey/Bag-of-Queries  (torch.hub get_trained_boq)
"""

from __future__ import annotations

import torch
import torchvision.transforms as T

from avl.models.eigenplaces import _trust_hub_repos


DESCRIPTOR_DIM = 12288

# The authors' recommended DINOv2 input size (README): 322x322, ImageNet stats.
BOQ_TRANSFORM = T.Compose(
    [
        T.Resize((322, 322), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class _BoQDescriptor(torch.nn.Module):
    """The hub model returns (descriptor, attentions); keep the descriptor."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.model(images)
        return out[0] if isinstance(out, (tuple, list)) else out


def load_boq(device: torch.device) -> torch.nn.Module:
    _trust_hub_repos("amaralibey/bag-of-queries")
    model = torch.hub.load(
        "amaralibey/bag-of-queries",
        "get_trained_boq",
        backbone_name="dinov2",
        output_dim=DESCRIPTOR_DIM,
        trust_repo=True,
    )
    return _BoQDescriptor(model).to(device).eval()
