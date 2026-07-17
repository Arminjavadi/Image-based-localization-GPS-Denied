from __future__ import annotations

import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError


REPO_ID = "Bancie/UAV-Self-Positioning-23M-ZCN"
CHECKPOINT_NAME = "UAV_SelfPositioning_23M_ZCN.pth"


class DenseUAVViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,
        )
        self.projection = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(images)
        local_tokens = tokens[:, 1:]
        pooled = local_tokens.mean(dim=1)
        return self.projection(pooled)


def load_denseuav_vit(device: torch.device) -> DenseUAVViT:
    try:
        checkpoint_path = hf_hub_download(
            REPO_ID,
            CHECKPOINT_NAME,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        checkpoint_path = hf_hub_download(REPO_ID, CHECKPOINT_NAME)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model = DenseUAVViT()
    backbone_prefix = "backbone.backbone."
    backbone_state = {
        key[len(backbone_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(backbone_prefix)
    }
    model.backbone.load_state_dict(backbone_state, strict=False)

    projection_state = {
        "0.weight": state_dict["head.head.classifier.add_block.0.weight"],
        "0.bias": state_dict["head.head.classifier.add_block.0.bias"],
        "1.weight": state_dict["head.head.classifier.add_block.1.weight"],
        "1.bias": state_dict["head.head.classifier.add_block.1.bias"],
        "1.running_mean": state_dict["head.head.classifier.add_block.1.running_mean"],
        "1.running_var": state_dict["head.head.classifier.add_block.1.running_var"],
        "1.num_batches_tracked": state_dict[
            "head.head.classifier.add_block.1.num_batches_tracked"
        ],
    }
    model.projection.load_state_dict(projection_state)
    return model.to(device).eval()


DENSEUAV_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
