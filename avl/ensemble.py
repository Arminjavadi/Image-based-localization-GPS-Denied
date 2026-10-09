"""Encoder ensembles — several encoders voting as one descriptor.

Each member's descriptor is L2-normalised, the members are concatenated, and the
result is scaled by 1/sqrt(n) so it stays unit length. The cosine of two ensemble
descriptors is then exactly the mean of the members' cosines:

    score = (cos_1 + cos_2 + ... + cos_n) / n

Nothing is trained and every member has an equal vote. On UAV-VisLoc r05 the
MegaLoc + Game4Loc + AnyLoc-L ensemble lifted top-1 within 100 m from 59.7 % (best
single encoder) to 80.6 %: encoders trained on different data fail on different
frames (docs/AVL_Progress_Report §5.5). The cost is the sum of the members.

Members keep their own reference / query caches and AnyLoc vocabularies, keyed by
the member name (see :func:`member_path`), so an ensemble run reuses the descriptors
the single-encoder runs already computed.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from avl.config import AVLConfig
from avl.encoder import VPREncoder
from avl.pipeline import join_model_spec, split_model_spec

# Native descriptor width per encoder; 0 = decided by the map-fitted AnyLoc vocabulary.
NATIVE_DIM = {
    "denseuav-vit": 512,
    "game4loc": 768,
    "sample4geo": 1024,
    "anyloc-lite": 0,
    "anyloc-l": 0,
    "anyloc-g": 0,
    "anyloc-sat": 0,
    "megaloc": 8448,
    "boq": 12288,
    "dinov3-b": 768,
    "dinov3-l": 1024,
    "infogeo": 4096,
    "infogeo-dense": 4096,
    "dinov2": 768,
    "dinov2-ft": 512,
    "mixvpr": 4096,
    "cosplace": 2048,
    "eigenplaces": 2048,
}


def validate_model_spec(spec: str) -> list[str]:
    members = split_model_spec(spec)
    unknown = [m for m in members if m not in NATIVE_DIM]
    if unknown:
        raise ValueError(f"unknown encoder(s) {unknown}; choose from {sorted(NATIVE_DIM)}")
    return members


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def combine_blocks(parts: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate per-member descriptors (each ``(..., d_i)``) into one ensemble
    descriptor whose cosine is the mean of the members' cosines. A single member is
    returned unchanged, so a one-encoder run is bit-identical to the plain encoder."""
    if len(parts) == 1:
        return np.ascontiguousarray(parts[0], dtype=np.float32)
    scale = 1.0 / np.sqrt(len(parts))
    out = np.concatenate([_l2n(np.asarray(p, dtype=np.float32)) * scale for p in parts], axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


def member_path(path: Path | None, spec: str, member: str) -> Path | None:
    """Per-member variant of a cache / vocabulary path written for ``spec``.

    ``cache_r05_t250__megaloc+game4loc.npy`` -> ``cache_r05_t250__game4loc.npy``, the
    file the single-encoder run already uses. For a single encoder the path is
    returned as given.
    """
    if path is None:
        return None
    path = Path(path)
    if spec == member:
        return path
    if spec in path.name:
        return path.with_name(path.name.replace(spec, member))
    return path.with_name(f"{path.stem}__{member}{path.suffix}")


class EnsembleEncoder:
    """Several :class:`~avl.encoder.VPREncoder` s behind the VPREncoder interface."""

    def __init__(self, members: Sequence[VPREncoder]) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = list(members)
        self.device = self.members[0].device
        self.config = copy.copy(self.members[0].config)
        self.config.model = join_model_spec([m.config.model for m in self.members])
        self.config.descriptor_dim = sum(self.blocks)

    @property
    def names(self) -> list[str]:
        return [m.config.model for m in self.members]

    @property
    def blocks(self) -> list[int]:
        return [int(m.config.descriptor_dim) for m in self.members]

    def encode_images(self, images: list[Image.Image], batch_size: int | None = None) -> np.ndarray:
        return combine_blocks([m.encode_images(images, batch_size) for m in self.members])

    def encode_paths(self, image_paths: Sequence[str | Path], show_progress: bool = True) -> np.ndarray:
        return combine_blocks([m.encode_paths(image_paths, show_progress) for m in self.members])


def members_of(encoder: VPREncoder | EnsembleEncoder) -> list[VPREncoder]:
    return list(encoder.members) if isinstance(encoder, EnsembleEncoder) else [encoder]


def blocks_of(encoder: VPREncoder | EnsembleEncoder) -> list[int] | None:
    """Member widths for block-wise centering; None for a single encoder."""
    return encoder.blocks if isinstance(encoder, EnsembleEncoder) and len(encoder.members) > 1 else None


def build_member(
    model: str,
    base: AVLConfig | None = None,
    vocab_path: Path | None = None,
    vocab_images: Sequence[str | Path] | None = None,
    descriptor_dim: int | None = None,
) -> VPREncoder:
    """One encoder, at its native width unless ``descriptor_dim`` says otherwise
    (MixVPR / CosPlace come in several). AnyLoc's vocabulary is map-specific: it is
    loaded from ``vocab_path`` when that exists, otherwise fitted on ``vocab_images``
    (the reference map, available before flight) and saved there."""
    config = copy.copy(base) if base is not None else AVLConfig(index_type="flat")
    config.model = model
    config.descriptor_dim = descriptor_dim or NATIVE_DIM[model]
    encoder = VPREncoder(config)
    if model.startswith("anyloc"):
        from avl.models.anyloc import fit_vocabulary

        if vocab_path is not None and Path(vocab_path).exists():
            encoder.model.load(Path(vocab_path))
        else:
            if vocab_images is None:
                raise ValueError(f"{model} needs a vocabulary file or the map images to fit one")
            fit_vocabulary(encoder.model, [str(p) for p in vocab_images], encoder.transform, encoder.device)
            if vocab_path is not None:
                Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
                encoder.model.save(Path(vocab_path))
        config.descriptor_dim = encoder.model.output_dim
        encoder.config = config
    return encoder


def build_encoder(
    spec: str,
    base: AVLConfig | None = None,
    vocab_path: Path | None = None,
    vocab_images: Sequence[str | Path] | None = None,
    descriptor_dim: int | None = None,
) -> VPREncoder | EnsembleEncoder:
    """A single encoder, or an :class:`EnsembleEncoder` for a ``a+b+c`` spec.
    ``vocab_path`` is written for ``spec``; members get :func:`member_path` variants.
    ``descriptor_dim`` applies to a single encoder only."""
    members = validate_model_spec(spec)
    if descriptor_dim and len(members) > 1:
        raise ValueError("descriptor_dim cannot be set for an ensemble")
    built = [
        build_member(m, base, member_path(vocab_path, spec, m), vocab_images, descriptor_dim)
        for m in members
    ]
    return built[0] if len(built) == 1 else EnsembleEncoder(built)
