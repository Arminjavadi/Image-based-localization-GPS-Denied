from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ModelName = Literal[
    "denseuav-vit", "game4loc", "sample4geo",
    "anyloc-lite", "anyloc-l", "anyloc-g", "anyloc-sat", "megaloc", "boq",
    "dinov3-b", "dinov3-l", "infogeo", "infogeo-dense",
    "dinov2", "dinov2-ft", "mixvpr", "cosplace", "eigenplaces",
]
IndexType = Literal["hnsw", "ivfpq", "flat"]


@dataclass
class AVLConfig:
    """Runtime configuration for AVL indexing and localization.

    The recipe fields (heading, centering, altitude scale, search window) are off by
    default so existing indexes and scripts behave as before;
    ``AVLConfig.from_preset("recommended")`` gives the pipeline the progress report
    shipped (avl.pipeline.PRESETS).
    """

    #: one encoder, or an ensemble "a+b+c" (avl.ensemble)
    model: ModelName | str = "denseuav-vit"
    descriptor_dim: int = 512
    cosplace_backbone: str = "ResNet101"
    # dinov2-ft only: the trained retrieval head (scripts/finetune_head.py).
    head_weights: Path = Path("artifacts/finetune/head.pt")
    device: str = "cuda"
    batch_size: int = 16
    query_rotations: int = 4

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

    # Geometric re-ranking: retrieval returns `rerank_candidates` tiles, local
    # features + RANSAC verify each one, and the inlier count re-orders them
    # before geo fusion. Off by default so existing runs are unchanged; see
    # avl.rerank for the backends and what each costs.
    rerank_enabled: bool = False
    rerank_backend: str = "sift-ransac"
    rerank_candidates: int = 10
    rerank_max_features: int = 2048
    rerank_image_size: int = 640
    rerank_ratio_test: float = 0.8
    rerank_ransac_threshold_px: float = 4.0
    rerank_min_inliers: int = 12
    # 0 keeps the descriptor order, 1 ranks purely on geometry.
    rerank_blend: float = 0.5

    # Geo-fusion: how the top-K matches are collapsed into one pose. See
    # avl.geo.FUSION_METHODS. "cluster" keeps the largest geo-consistent group of
    # matches and drops the outliers that otherwise drag a plain average off.
    fusion_method: str = "cluster"
    fusion_softmax_temp: float = 0.02
    fusion_cluster_radius_m: float = 150.0

    # -- recipe: the measured, training-free steps (avl.pipeline) ----------------
    #: off | snap90 | exact | auto — turn the frame north-up from the heading.
    #: A frame without a heading falls back to the four-rotation search.
    heading_mode: str = "off"
    #: off | map | map+flight — subtract the map's / the flight's mean descriptor.
    #: The map mean is computed when the index is built and stored with it.
    center_mode: str = "off"
    center_warmup: int = 10
    #: crop the frame to the tile's ground size from height above ground
    scale_from_agl: bool = False
    #: camera constant: full-frame ground width / AGL = 2 tan(HFOV / 2)
    camera_k: float | None = None
    #: reference tile ground size (m); parsed from '_t<metres>' in the map path if unset
    tile_m: float | None = None
    #: crop only frames covering more than this x a tile
    agl_gate: float | None = 1.25
    min_agl_m: float = 100.0
    dem_dir: Path = Path("data/dem/copernicus_glo30")
    #: search-window radius in prior sigmas (the sigma comes with each query)
    prior_k: float = 3.0
    #: none | square — only used for frames that are not north-aligned
    query_crop: str = "none"

    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "avl")

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.dem_dir = Path(self.dem_dir)
        if self.query_rotations not in {1, 4}:
            raise ValueError("query_rotations must be 1 or 4")
        from avl.centering import CENTER_MODES
        from avl.geo import FUSION_METHODS
        from avl.pipeline import HEADING_MODES

        if self.fusion_method not in FUSION_METHODS:
            raise ValueError(f"fusion_method must be one of {FUSION_METHODS}")
        if self.heading_mode not in HEADING_MODES:
            raise ValueError(f"heading_mode must be one of {HEADING_MODES}")
        if self.center_mode not in CENTER_MODES:
            raise ValueError(f"center_mode must be one of {CENTER_MODES}")
        if self.query_crop not in {"none", "square"}:
            raise ValueError("query_crop must be 'none' or 'square'")

        from avl.rerank import RERANK_BACKENDS

        if self.rerank_backend not in RERANK_BACKENDS:
            raise ValueError(f"rerank_backend must be one of {RERANK_BACKENDS}")
        if not 0.0 <= self.rerank_blend <= 1.0:
            raise ValueError("rerank_blend must be between 0 and 1")
        if self.rerank_candidates < 1:
            raise ValueError("rerank_candidates must be >= 1")

    def rerank_search_k(self) -> int:
        """How many candidates retrieval must return to feed the configured stage."""
        if not self.rerank_enabled:
            return self.top_k
        return max(self.top_k, self.rerank_candidates)

    def apply_recipe(self, recipe) -> "AVLConfig":
        """Copy an :class:`avl.pipeline.Recipe` into these fields (returns self)."""
        self.model = recipe.model_spec
        self.query_rotations = recipe.effective_rotations
        self.heading_mode = recipe.heading
        self.center_mode = recipe.center
        self.scale_from_agl = bool(recipe.agl_scale)
        self.agl_gate = recipe.agl_gate
        if recipe.camera_k is not None:
            self.camera_k = recipe.camera_k
        self.prior_k = recipe.window_k
        self.fusion_method = recipe.fusion
        if recipe.center != "off":
            # Centred cosines of *correct* fixes run 0.13-0.4 (UAV-VisLoc r05 / r11), and
            # wrong matches score as high: an absolute gate would reject ~90 % of good fixes
            # without separating bad ones. Rejection is the navigation filter's χ² gate.
            self.score_threshold = 0.0
        self.query_crop = "none" if recipe.heading != "off" else recipe.query_crop
        return self

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "AVLConfig":
        """A config for one of avl.pipeline.PRESETS ('baseline', 'recommended',
        'accuracy'), with an exact (flat) index unless ``overrides`` say otherwise."""
        from avl.pipeline import PRESETS

        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        config = cls(index_type="flat").apply_recipe(PRESETS[name][2])
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise TypeError(f"AVLConfig has no field {key!r}")
            setattr(config, key, value)
        config.__post_init__()
        return config
