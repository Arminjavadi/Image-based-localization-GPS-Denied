"""Trainable retrieval head on frozen DINOv2 tokens (see avl.finetune.backbone).

Two aggregators:

gem  per-token MLP, GeM pooling over the 8 x 8 grid, concatenated with a projected
     CLS token. Orderless: robust to the residual (< 45 deg) rotation left after the
     four-orientation search, blind to layout.
mix  per-token MLP to a narrow width, then the 8 x 8 grid flattened and projected
     (MixVPR-style). Keeps spatial layout, which is what separates neighbouring
     tiles; relies on the four-orientation search for rotation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from avl.finetune.backbone import EMBED_DIM, GRID


class GeM(nn.Module):
    def __init__(self, p: float = 3.0) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, C) -> (B, C)
        p = self.p.clamp(1.0, 8.0)
        return x.clamp(min=1e-6).pow(p).mean(dim=1).pow(1.0 / p)


class RetrievalHead(nn.Module):
    def __init__(self, kind: str = "gem", hidden: int = 512, out_dim: int = 512,
                 mix_width: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.kind = kind
        self.config = {"kind": kind, "hidden": hidden, "out_dim": out_dim,
                       "mix_width": mix_width, "dropout": dropout}
        self.norm = nn.LayerNorm(EMBED_DIM)
        width = hidden if kind == "gem" else mix_width
        self.token_mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width)
        )
        if kind == "gem":
            self.pool = GeM()
            self.cls_proj = nn.Linear(EMBED_DIM, hidden)
            self.out = nn.Linear(2 * hidden, out_dim)
        elif kind == "mix":
            self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(GRID * GRID * width, out_dim))
        else:
            raise ValueError(f"unknown head kind {kind!r}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (..., 65, 768) -> (..., out_dim), L2-normalised."""
        lead = tokens.shape[:-2]
        x = self.norm(tokens.reshape(-1, *tokens.shape[-2:]).float())
        cls, grid = x[:, 0], x[:, 1:]
        g = self.token_mlp(grid)
        if self.kind == "gem":
            d = self.out(torch.cat([self.pool(F.relu(g)), self.cls_proj(cls)], dim=1))
        else:
            d = self.out(g.flatten(1))
        return F.normalize(d, dim=-1).reshape(*lead, -1)

    def save(self, path: Path, **meta) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path: Path | str, map_location: str = "cpu") -> "RetrievalHead":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        head = cls(**ckpt["config"])
        head.load_state_dict(ckpt["state_dict"])
        return head.eval()
