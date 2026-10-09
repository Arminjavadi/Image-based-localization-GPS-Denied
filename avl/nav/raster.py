"""UAV-VisLoc region mosaics: geo <-> pixel mapping and a downsampled thumbnail.

Each region ships one north-up georeferenced satellite mosaic
(`data/UAVVisLoc/<r>/satellite<r>.tif`) whose corner lat/lons are listed in
`data/UAVVisLoc/satellite_ coordinates_range.csv` (LT = left-top pixel, RB =
right-bottom pixel; longitude increases left->right, latitude decreases
top->bottom). This module wraps that transform and provides the drawn-path ->
nearest-frames snap used by the "Trajectory Studio" GUI tab.

numpy only, plus Pillow to read the mosaic (already a project dependency).
Qt-free: the GUI converts the returned PIL image to a QPixmap itself.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # Pillow is a hard project dep; guard only so `import avl.nav` never explodes
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
except Exception:  # pragma: no cover
    Image = None  # type: ignore

RANGES_CSV = Path("data/UAVVisLoc/satellite_ coordinates_range.csv")


@dataclass(frozen=True)
class RegionRaster:
    region: str
    tif_path: Path
    lt_lat: float
    lt_lon: float
    rb_lat: float
    rb_lon: float
    width: int
    height: int

    # -- discovery ----------------------------------------------------
    @classmethod
    def load_ranges(cls, project_root: Path) -> dict[str, tuple]:
        path = project_root / RANGES_CSV
        out: dict[str, tuple] = {}
        if not path.is_file():
            return out
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                name = row["mapname"]
                region = "".join(ch for ch in name if ch.isdigit())
                out[region] = (
                    name,
                    float(row["LT_lat_map"]),
                    float(row["LT_lon_map"]),
                    float(row["RB_lat_map"]),
                    float(row["RB_lon_map"]),
                )
        return out

    @classmethod
    def for_region(cls, region: str, project_root: Path) -> "RegionRaster | None":
        region = str(region).lstrip("r").zfill(2)
        entry = cls.load_ranges(project_root).get(region)
        if entry is None or Image is None:
            return None
        name, lt_lat, lt_lon, rb_lat, rb_lon = entry
        tif = project_root / "data" / "UAVVisLoc" / region / name
        if not tif.is_file():
            return None
        with Image.open(tif) as im:
            w, h = im.size
        return cls(region, tif, lt_lat, lt_lon, rb_lat, rb_lon, int(w), int(h))

    @classmethod
    def discover(cls, project_root: Path) -> dict[str, "RegionRaster"]:
        found: dict[str, RegionRaster] = {}
        for region in cls.load_ranges(project_root):
            raster = cls.for_region(region, project_root)
            if raster is not None:
                found[region] = raster
        return found

    # -- geo <-> full-resolution pixel -------------------------------
    def geo_to_px(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        x = (lon - self.lt_lon) / (self.rb_lon - self.lt_lon) * self.width
        y = (lat - self.lt_lat) / (self.rb_lat - self.lt_lat) * self.height
        return x, y

    def px_to_geo(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        lon = self.lt_lon + x / self.width * (self.rb_lon - self.lt_lon)
        lat = self.lt_lat + y / self.height * (self.rb_lat - self.lt_lat)
        return lat, lon

    # -- thumbnail --------------------------------------------------
    def thumbnail(self, max_px: int = 2000):
        """Return (PIL.Image RGB, scale) where scale = thumb_px / full_px."""
        if Image is None:  # pragma: no cover
            raise RuntimeError("Pillow is required to render the mosaic thumbnail")
        scale = min(1.0, max_px / max(self.width, self.height))
        size = (max(1, round(self.width * scale)), max(1, round(self.height * scale)))
        with Image.open(self.tif_path) as im:
            thumb = im.convert("RGB").resize(size, Image.BILINEAR)
        return thumb, scale

    def as_dict(self) -> dict:
        return {
            "region": self.region,
            "path": str(self.tif_path),
            "lt": [self.lt_lat, self.lt_lon],
            "rb": [self.rb_lat, self.rb_lon],
            "size": [self.width, self.height],
        }


# --------------------------------------------------------------------------- #
# drawn path  ->  nearest real frames, in order
# --------------------------------------------------------------------------- #
def _point_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Distance from ``pt`` to segment ``a-b`` and the along-segment length of the foot."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.hypot(*(pt - a))), 0.0
    t = float(np.clip((pt - a) @ ab / denom, 0.0, 1.0))
    foot = a + t * ab
    return float(np.hypot(*(pt - foot))), t * float(np.hypot(*ab))


def _dist_to_polyline(frame_xy: np.ndarray, path_xy: np.ndarray) -> np.ndarray:
    """Min distance of every point in ``frame_xy`` (N,2) to the polyline (V,2)."""
    out = np.full(len(frame_xy), np.inf)
    for i, pt in enumerate(frame_xy):
        for j in range(len(path_xy) - 1):
            d, _ = _point_segment(pt, path_xy[j], path_xy[j + 1])
            if d < out[i]:
                out[i] = d
    return out


def snap_path_to_frames(
    path_xy: np.ndarray,
    frame_xy: np.ndarray,
    frame_ids: list[str],
    corridor_m: float,
) -> list[str]:
    """Frames within ``corridor_m`` of the polyline, ordered by arc-length along it.

    All coordinates are a common planar frame (local ENU metres). ``path_xy`` is
    (V, 2), ``frame_xy`` is (N, 2). This is the "path order" (re-ordering) mode —
    for a physically meaningful segment use :func:`contiguous_run_along_path`.
    """
    path_xy = np.asarray(path_xy, dtype=np.float64)
    frame_xy = np.asarray(frame_xy, dtype=np.float64)
    if len(path_xy) < 2:
        return []

    seg_start_len = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path_xy, axis=0).T))])
    picked: list[tuple[float, str]] = []
    for i, pt in enumerate(frame_xy):
        best_d, best_s = np.inf, 0.0
        for j in range(len(path_xy) - 1):
            d, along = _point_segment(pt, path_xy[j], path_xy[j + 1])
            if d < best_d:
                best_d, best_s = d, seg_start_len[j] + along
        if best_d <= corridor_m:
            picked.append((best_s, frame_ids[i]))
    picked.sort(key=lambda t: t[0])
    return [fid for _, fid in picked]


def contiguous_run_along_path(
    path_xy: np.ndarray,
    frame_xy: np.ndarray,
    corridor_m: float,
    max_gap: int = 3,
) -> tuple[int, int, float] | None:
    """The longest run of consecutive frames that stays within ``corridor_m`` of
    the polyline (bridging gaps of up to ``max_gap`` frames).

    ``frame_xy`` (N, 2) must be in recorded order. Returns
    ``(start, end, coverage)`` — inclusive frame indices and the fraction of
    frames in ``[start, end]`` actually inside the corridor — or ``None``.

    Unlike :func:`snap_path_to_frames`, the result is a real, spatially and
    temporally contiguous flight segment: the strapdown INS integrates it without
    the acceleration spikes a re-ordered, sparse selection would inject.
    """
    path_xy = np.asarray(path_xy, dtype=np.float64)
    frame_xy = np.asarray(frame_xy, dtype=np.float64)
    if len(path_xy) < 2 or len(frame_xy) == 0:
        return None

    inside = _dist_to_polyline(frame_xy, path_xy) <= corridor_m
    if not inside.any():
        return None

    best = (-1, -1, 0)
    start: int | None = None
    end = -1
    gap = 0
    for i, ok in enumerate(inside):
        if ok:
            if start is None:
                start = i
            end = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                if end - start + 1 > best[2]:
                    best = (start, end, end - start + 1)
                start, gap = None, 0
    if start is not None and end - start + 1 > best[2]:
        best = (start, end, end - start + 1)

    s, e, _ = best
    if s < 0:
        return None
    coverage = float(inside[s : e + 1].mean())
    return s, e, coverage
