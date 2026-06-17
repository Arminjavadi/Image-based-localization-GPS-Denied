from __future__ import annotations

import torch
import torchvision.transforms as T

COSPLACE_TRANSFORM = T.Compose(
    [
        T.Resize((322, 322), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_cosplace(backbone: str, descriptor_dim: int, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(
        "gmberton/CosPlace",
        "get_trained_model",
        backbone=backbone,
        fc_output_dim=descriptor_dim,
        trust_repo=True,
    )
    return model.to(device).eval()
