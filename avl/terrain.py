"""Terrain elevation and height above ground (AGL) for scale normalisation.

A drone frame's ground footprint is proportional to its height *above the ground*,
not above sea level. Flight logs (and UAV-VisLoc's ``height`` column) give altitude
above sea level, so the terrain under the vehicle has to be subtracted:

    AGL = altitude_ASL - terrain_elevation(lat, lon)

The terrain comes from the Copernicus GLO-30 DEM: global, 1 arc-second (~30 m),
free, served as one Cloud-Optimised GeoTIFF per 1 x 1 degree cell from the AWS
open-data bucket. Heights are orthometric (EGM2008 geoid). :func:`ensure_tiles`
downloads the cells a map needs once; afterwards everything runs offline.

In flight, the terrain is looked up at the *estimated* position (the navigation
prior), not the true one. The DEM is smooth at the scale of a prior's error except
in steep terrain, which is why this works without knowing where you are exactly.
"""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

COPERNICUS_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
)
DEFAULT_DEM_DIR = Path("data/dem/copernicus_glo30")


def tile_name(lat_floor: int, lon_floor: int) -> str:
    """Copernicus GLO-30 cell name for the 1-degree cell whose south-west corner is given."""
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_"
        f"{ew}{abs(lon_floor):03d}_00_DEM"
    )


def cells_for_bounds(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list[tuple[int, int]]:
    return [
        (la, lo)
        for la in range(math.floor(lat_min), math.floor(lat_max) + 1)
        for lo in range(math.floor(lon_min), math.floor(lon_max) + 1)
    ]


def ensure_tiles(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    dem_dir: Path = DEFAULT_DEM_DIR,
    timeout_s: float = 120.0,
) -> list[Path]:
    """Download (once) every DEM cell covering the bounds. Returns the local paths.

    A cell missing from the bucket (open ocean) is skipped; lookups there return NaN.
    """
    dem_dir = Path(dem_dir)
    dem_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for la, lo in cells_for_bounds(lat_min, lat_max, lon_min, lon_max):
        name = tile_name(la, lo)
        path = dem_dir / f"{name}.tif"
        if not path.exists():
            tmp = path.with_suffix(".part")
            try:
                with urllib.request.urlopen(COPERNICUS_URL.format(name=name), timeout=timeout_s) as r:
                    tmp.write_bytes(r.read())
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 404):
                    continue  # no land in this cell
                raise
            tmp.rename(path)
        paths.append(path)
    return paths


class _Cell:
    """One decoded DEM cell with its pixel-to-geo transform."""

    def __init__(self, path: Path) -> None:
        try:
            import tifffile
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "reading the DEM needs tifffile + imagecodecs: pip install -e '.[terrain]'"
            ) from exc
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            self.z = page.asarray().astype(np.float32)
            tags = page.tags
            sx, sy, _ = tags["ModelPixelScaleTag"].value
            tie = tags["ModelTiepointTag"].value  # (i, j, k, x, y, z)
            geokeys = tif.geotiff_metadata or {}
        # GeoTIFF raster type 2 = PixelIsPoint: the tiepoint is a pixel *centre*.
        # Type 1 = PixelIsArea: it is the pixel's top-left corner.
        raster_type = int(geokeys.get("GTRasterTypeGeoKey", 1))  # IntEnum in tifffile
        half = 0.0 if raster_type == 2 else 0.5
        self.lon0 = float(tie[3]) - float(tie[0]) * sx  # lon of column 0 (centre or edge)
        self.lat0 = float(tie[4]) + float(tie[1]) * sy
        self.dlon, self.dlat = float(sx), float(sy)
        self.half = half
        rows, cols = self.z.shape
        self.lat_top = self.lat0 + (half * self.dlat)
        self.lon_left = self.lon0 - (half * self.dlon)
        self.lat_bottom = self.lat_top - rows * self.dlat
        self.lon_right = self.lon_left + cols * self.dlon

    def covers(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return (lat <= self.lat_top) & (lat >= self.lat_bottom) & (lon >= self.lon_left) & (lon <= self.lon_right)

    def sample(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Bilinear interpolation between pixel centres."""
        # continuous pixel coordinates of the query, in pixel-centre units
        col = (lon - self.lon0) / self.dlon - self.half
        row = (self.lat0 - lat) / self.dlat - self.half
        rows, cols = self.z.shape
        c0 = np.clip(np.floor(col).astype(int), 0, cols - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, rows - 2)
        fc = np.clip(col - c0, 0.0, 1.0)
        fr = np.clip(row - r0, 0.0, 1.0)
        z = self.z
        top = z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc
        bot = z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc
        return top * (1 - fr) + bot * fr


class TerrainModel:
    """Terrain elevation (m, EGM2008) from Copernicus GLO-30 cells on disk."""

    def __init__(self, dem_dir: Path = DEFAULT_DEM_DIR) -> None:
        self.dem_dir = Path(dem_dir)
        self._cells: dict[Path, _Cell] = {}

    @classmethod
    def for_bounds(
        cls,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        dem_dir: Path = DEFAULT_DEM_DIR,
        download: bool = True,
    ) -> "TerrainModel":
        model = cls(dem_dir)
        if download:
            ensure_tiles(lat_min, lat_max, lon_min, lon_max, dem_dir)
        return model

    def _cell(self, lat_floor: int, lon_floor: int) -> _Cell | None:
        path = self.dem_dir / f"{tile_name(lat_floor, lon_floor)}.tif"
        if not path.exists():
            return None
        if path not in self._cells:
            self._cells[path] = _Cell(path)
        return self._cells[path]

    def elevation(self, lat, lon) -> np.ndarray:
        """Terrain height at each (lat, lon); NaN where no DEM cell is on disk."""
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        out = np.full(lat.shape, np.nan, dtype=np.float64)
        keys = np.stack([np.floor(lat), np.floor(lon)], axis=1).astype(int)
        for la, lo in {tuple(k) for k in keys}:
            sel = (keys[:, 0] == la) & (keys[:, 1] == lo)
            cell = self._cell(la, lo)
            if cell is not None:
                out[sel] = cell.sample(lat[sel], lon[sel])
        return out

    def agl(self, altitude_asl, lat, lon) -> np.ndarray:
        """Height above ground = altitude above sea level minus terrain elevation."""
        return np.asarray(altitude_asl, dtype=np.float64) - self.elevation(lat, lon)
