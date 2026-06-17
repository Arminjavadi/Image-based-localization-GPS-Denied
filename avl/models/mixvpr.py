from __future__ import annotations

import os
from pathlib import Path

import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

MODELS_INFO = {
    128: (
        "https://drive.google.com/uc?id=1DQnefjk1hVICOEYPwE4-CZAZOvi1NSJz",
        "resnet50_MixVPR_128_channels(64)_rows(2)",
        64,
        2,
    ),
    512: (
        "https://drive.google.com/uc?id=1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj",
        "resnet50_MixVPR_512_channels(256)_rows(2)",
        256,
        2,
    ),
    4096: (
        "https://drive.google.com/uc?id=1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L",
        "resnet50_MixVPR_4096_channels(1024)_rows(4)",
        1024,
        4,
    ),
}


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim: int, mlp_ratio: float = 1) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(x)


class MixVPRAggregator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        in_h: int = 20,
        in_w: int = 20,
        out_channels: int = 512,
        mix_depth: int = 1,
        mlp_ratio: float = 1,
        out_rows: int = 4,
    ) -> None:
        super().__init__()
        hw = in_h * in_w
        self.mix = nn.Sequential(
            *[FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio) for _ in range(mix_depth)]
        )
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        return F.normalize(x.flatten(1), p=2, dim=-1)


class MixVPRBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = torchvision.models.resnet50(weights=None)
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        backbone.layer4 = nn.Identity()
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x


class MixVPRNet(nn.Module):
    def __init__(self, agg_config: dict) -> None:
        super().__init__()
        self.backbone = MixVPRBackbone()
        self.aggregator = MixVPRAggregator(**agg_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.aggregator(x)


def load_mixvpr(descriptors_dimension: int, cache_dir: Path, device: torch.device) -> MixVPRNet:
    if descriptors_dimension not in MODELS_INFO:
        supported = sorted(MODELS_INFO)
        raise ValueError(f"MixVPR descriptor dim must be one of {supported}, got {descriptors_dimension}")

    url, filename, out_channels, out_rows = MODELS_INFO[descriptors_dimension]
    model_config = {
        "in_channels": 1024,
        "in_h": 20,
        "in_w": 20,
        "out_channels": out_channels,
        "mix_depth": 4,
        "mlp_ratio": 1,
        "out_rows": out_rows,
    }
    model = MixVPRNet(agg_config=model_config)

    weights_dir = cache_dir / "mixvpr"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / filename
    if not weights_path.exists():
        gdown.download(url=url, output=str(weights_path), quiet=False)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


MIXVPR_TRANSFORM = T.Compose(
    [
        T.Resize((320, 320), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
