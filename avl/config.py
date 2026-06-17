from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ModelName = Literal["mixvpr", "cosplace"]
IndexType = Literal["hnsw", "ivfpq", "flat"]


@dataclass
class AVLConfig:
    """Runtime configuration for AVL indexing and localization."""

    model: ModelName = "mixvpr"
    descriptor_dim: int = 4096
    cosplace_backbone: str = "ResNet101"
    device: str = "cuda"
    batch_size: int = 16

    index_type: IndexType = "hnsw"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    ivfpq_nlist: int = 4096
    ivfpq_m: int = 64
    ivfpq_nbits: int = 8
    ivfpq_train_samples: int = 256_000

    top_k: int = 5
    score_threshold: float = 0.35
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "avl")

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
