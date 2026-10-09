"""Domain centering — remove each image domain's shared descriptor component.

Every descriptor from one source (the drone camera, the satellite provider) carries
a large component common to all images from that source: colour response, blur,
sun angle, season, processing. It is the same for every tile, so it carries no
place information, but it dominates the cosine score and squeezes the margin
between the right tile and the rest. Subtracting the domain's mean descriptor
and re-normalising leaves the part that tells places apart.

Modes (nothing is trained, no labels are used):

  off         raw descriptors (the historical behaviour)
  map         subtract the reference map's mean from both sides. The mean is
              computed once when the map is encoded, so a query pays one vector
              subtraction. Works for a single image.
  map+flight  refs centred on the map mean; each query centred on the running
              mean of the flight frames seen so far (causal — never uses future
              frames). The first ``warmup`` frames blend toward the map mean
              while the flight mean is still noisy.

For an encoder ensemble (avl.ensemble) pass the member widths as ``blocks``: each
member is centred and re-normalised on its own and the result is scaled back to the
ensemble's equal-vote layout, so centering cannot let one member outvote the others.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

CENTER_MODES = ("off", "map", "map+flight")
DEFAULT_WARMUP = 10


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _l2n_blocks(x: np.ndarray, blocks: Sequence[int] | None) -> np.ndarray:
    if not blocks or len(blocks) < 2:
        return _l2n(x)
    edges = np.cumsum([0, *blocks])
    if edges[-1] != x.shape[-1]:
        raise ValueError(f"blocks {list(blocks)} do not add up to descriptor width {x.shape[-1]}")
    scale = 1.0 / np.sqrt(len(blocks))
    return np.concatenate(
        [_l2n(x[..., a:b]) * scale for a, b in zip(edges[:-1], edges[1:])], axis=-1
    )


class DomainCentering:
    def __init__(
        self, mode: str = "off", warmup: int = DEFAULT_WARMUP, blocks: Sequence[int] | None = None
    ) -> None:
        if mode not in CENTER_MODES:
            raise ValueError(f"unknown centering mode {mode!r}; choose from {CENTER_MODES}")
        self.mode = mode
        self.warmup = max(1, int(warmup))
        self.blocks = list(blocks) if blocks else None
        self.map_mean: np.ndarray | None = None
        self._flight_sum: np.ndarray | None = None
        self._flight_n = 0

    @property
    def frames_seen(self) -> int:
        return self._flight_n

    def fit_refs(self, ref_desc: np.ndarray) -> np.ndarray:
        """Store the map mean and return the reference descriptors to index."""
        if self.mode == "off":
            return ref_desc
        self.map_mean = ref_desc.mean(axis=0)
        return np.ascontiguousarray(_l2n_blocks(ref_desc - self.map_mean, self.blocks), dtype=np.float32)

    def reset_flight(self) -> None:
        """Forget the flight mean (a new flight over the same map)."""
        self._flight_sum = None
        self._flight_n = 0

    def query(self, desc: np.ndarray) -> np.ndarray:
        """Centre one frame's descriptors (variants x dim); updates the flight mean."""
        if self.mode == "off":
            return desc
        if self.map_mean is None:
            raise RuntimeError("fit_refs() must run before query()")
        mean = self.map_mean
        if self.mode == "map+flight":
            frame = desc.mean(axis=0)
            self._flight_sum = frame if self._flight_sum is None else self._flight_sum + frame
            self._flight_n += 1
            mean = self._flight_sum / self._flight_n
            if self._flight_n < self.warmup:
                a = self._flight_n / self.warmup
                mean = a * mean + (1.0 - a) * self.map_mean
        return _l2n_blocks(desc - mean, self.blocks).astype(np.float32)
