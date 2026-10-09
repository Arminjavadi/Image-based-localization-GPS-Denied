"""Scale normalisation — make each drone frame cover the same ground as a map tile.

Retrieval compares a drone frame with square reference tiles of a fixed ground size
(``tile_m``). The frame's own ground coverage changes with height above ground: a
frame taken 600 m above a valley covers twice the ground of one taken 300 m above a
ridge, yet both are resized to the same encoder input. This module turns height
above ground (see :mod:`avl.terrain`) into the centre-crop fraction that makes the
query cover exactly ``tile_m`` metres, so every frame is compared at the map's scale.

Pinhole nadir model: the full frame's ground width is ``k * AGL`` with
``k = 2 * tan(HFOV / 2)``, so the ground sampling distance is
``gsd = k * AGL / width_px`` metres per pixel. ``k`` is a property of the camera
(calibrated once: from the lens datasheet, or from a handful of frames matched to
the map — see ``scripts/visloc_scale.py calibrate``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraFootprint:
    """Ground footprint of a nadir camera: full-frame width = ``k`` x AGL."""

    k: float
    width_px: int

    @classmethod
    def from_hfov(cls, hfov_deg: float, width_px: int) -> "CameraFootprint":
        return cls(2.0 * math.tan(math.radians(hfov_deg) / 2.0), int(width_px))

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.k / 2.0))

    def gsd_m(self, agl_m: float, width_px: int | None = None) -> float:
        """Metres per pixel at ``agl_m``, for an image ``width_px`` wide (default: native)."""
        w = width_px or self.width_px
        return self.k * agl_m / w


def query_square_side_px(width: int, height: int, heading_deg: float | None) -> float:
    """Side of the square the retrieval pipeline actually encodes, in source pixels.

    Mirrors :func:`avl.retrieval.query_variants`: north-aligned frames keep the
    largest padding-free square after rotation (``rotate_no_padding``), whose side
    shrinks to ~0.71 x the short side at 45 deg; otherwise it is the centred
    short-side square. Ignoring this makes the ground scale depend on heading.
    """
    side = float(min(width, height))
    if heading_deg is None or not np.isfinite(heading_deg):
        return side
    theta = math.radians(heading_deg)
    return side / (abs(math.cos(theta)) + abs(math.sin(theta)))


def crop_fraction(side_px: float, gsd_m: float, target_ground_m: float) -> float:
    """Centre-crop fraction so the encoded square covers ``target_ground_m``.

    Values above 1 mean the frame covers *less* ground than a tile; cropping
    cannot fix that, so the fraction is clipped to 1 (use smaller tiles instead).
    """
    if not (np.isfinite(gsd_m) and gsd_m > 0 and side_px > 0):
        return 1.0
    return float(min(1.0, target_ground_m / (side_px * gsd_m)))


def query_scale(
    camera: CameraFootprint | None,
    agl_m: float | None,
    width: int,
    height: int,
    heading_deg: float | None,
    tile_m: float | None,
    min_agl_m: float = 100.0,
    gate: float | None = None,
) -> tuple[float, ...]:
    """The ``scales`` argument for :func:`avl.retrieval.query_variants` for one frame.

    One informed crop when camera, AGL and tile size are all known and the vehicle
    is airborne; otherwise the uncropped frame (the historical behaviour).

    ``gate``: crop only when the frame's footprint (:func:`frame_ground_m`) exceeds
    ``gate`` x the tile's ground.
    Cropping discards context, which costs more than a small scale mismatch: on
    UAV-VisLoc it lifted MegaLoc +37 pts where frames were 1.9x the tile and did
    nothing or hurt where they were within ~1.1-1.5x (docs/Scale_Normalization_Report.md).
    """
    if camera is None or tile_m is None or agl_m is None or not np.isfinite(agl_m) or agl_m < min_agl_m:
        return (1.0,)
    if gate is not None and frame_ground_m(camera, float(agl_m), width, height) <= gate * tile_m:
        return (1.0,)
    fraction, _ = agl_crop_fraction(camera, float(agl_m), width, height, heading_deg, tile_m)
    return (fraction,)


def frame_ground_m(camera: CameraFootprint, agl_m: float, width: int, height: int) -> float:
    """Ground side of the frame's full short-side square: the frame's footprint, before any
    rotation crop. This is what a tile should match (``visloc_scale.py plan``)."""
    return min(width, height) * camera.gsd_m(agl_m, width)


def agl_crop_fraction(
    camera: CameraFootprint,
    agl_m: float,
    width: int,
    height: int,
    heading_deg: float | None,
    tile_m: float,
) -> tuple[float, float]:
    """(crop fraction, ground side in metres of the uncropped square) for one frame."""
    gsd = camera.gsd_m(agl_m, width)
    side_px = query_square_side_px(width, height, heading_deg)
    return crop_fraction(side_px, gsd, tile_m), side_px * gsd
