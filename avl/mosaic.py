"""Windowed reads from UAV-VisLoc satellite mosaics.

The mosaics are uncompressed, tiled RGB GeoTIFFs of up to ~2.5 GB. PIL decodes the
whole raster on the first crop, which does not fit alongside training on a 16 GB
machine, so this reader memory-maps the file and assembles a crop from only the
256 x 256 storage tiles it touches. Region 09 is shipped as a 2 x 2 grid of such
files (``satelliteNN_RR-CC.tif``) under one georeference; :class:`RegionMosaic`
stitches them virtually.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat = math.radians(lat_deg)
    m_lat = 111_132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_lon = 111_412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return m_lat, m_lon


class TiledTiff:
    """Random-access reader for an uncompressed, chunky-RGB, tiled TIFF."""

    def __init__(self, path: Path) -> None:
        with Image.open(path) as im:
            if im.mode != "RGB" or not im.tile or im.tile[0][0] != "raw":
                raise ValueError(f"{path}: expected uncompressed tiled RGB, got {im.mode} {im.tile[:1]}")
            self.width, self.height = im.size
            self.tw = int(im.tag_v2[322])  # TileWidth
            self.th = int(im.tag_v2[323])  # TileLength
            self.offsets = {(t[1][0] // self.tw, t[1][1] // self.th): int(t[2]) for t in im.tile}
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def _tile(self, cx: int, cy: int) -> np.ndarray:
        off = self.offsets[(cx, cy)]
        return self.mm[off : off + self.tw * self.th * 3].reshape(self.th, self.tw, 3)

    def read(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """Pixels [y0:y1, x0:x1]; out-of-raster pixels are black."""
        out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        cx0, cx1 = max(0, x0) // self.tw, (min(x1, self.width) - 1) // self.tw
        cy0, cy1 = max(0, y0) // self.th, (min(y1, self.height) - 1) // self.th
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                tx, ty = cx * self.tw, cy * self.th
                ax0, ay0 = max(x0, tx), max(y0, ty)
                ax1 = min(x1, tx + self.tw, self.width)
                ay1 = min(y1, ty + self.th, self.height)
                if ax0 >= ax1 or ay0 >= ay1:
                    continue
                tile = self._tile(cx, cy)
                out[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] = tile[ay0 - ty : ay1 - ty, ax0 - tx : ax1 - tx]
        return out


class RegionMosaic:
    """A region's georeferenced satellite map, cropped by ground coordinates."""

    def __init__(self, root: Path, region: str) -> None:
        ranges = pd.read_csv(root / "satellite_ coordinates_range.csv")
        ranges["rid"] = ranges["mapname"].str.extract(r"satellite(\d+)")
        row = ranges[ranges["rid"] == region].iloc[0]
        self.lt_lat, self.lt_lon = float(row["LT_lat_map"]), float(row["LT_lon_map"])
        self.rb_lat, self.rb_lon = float(row["RB_lat_map"]), float(row["RB_lon_map"])

        single = root / region / row["mapname"]
        if single.exists():
            parts = {(0, 0): TiledTiff(single)}
        else:
            parts = {}
            for p in sorted((root / region).glob(f"satellite{region}_*.tif")):
                r, c = map(int, re.search(r"_(\d+)-(\d+)\.tif$", p.name).groups())
                parts[(r - 1, c - 1)] = TiledTiff(p)
            if not parts:
                raise FileNotFoundError(f"no satellite map for region {region}")
        # virtual layout: column widths from row 0, row heights from column 0
        n_r = 1 + max(r for r, _ in parts)
        n_c = 1 + max(c for _, c in parts)
        self.col_x = np.cumsum([0] + [parts[(0, c)].width for c in range(n_c)])
        self.row_y = np.cumsum([0] + [parts[(r, 0)].height for r in range(n_r)])
        self.parts = parts
        self.width, self.height = int(self.col_x[-1]), int(self.row_y[-1])

        self.dlat = (self.rb_lat - self.lt_lat) / self.height
        self.dlon = (self.rb_lon - self.lt_lon) / self.width
        m_lat, m_lon = meters_per_degree(0.5 * (self.lt_lat + self.rb_lat))
        self.gsd_x = abs(self.dlon) * m_lon
        self.gsd_y = abs(self.dlat) * m_lat

    def contains(self, lat: float, lon: float, margin_m: float = 0.0) -> bool:
        x, y = self.to_px(lat, lon)
        mx, my = margin_m / self.gsd_x, margin_m / self.gsd_y
        return mx <= x <= self.width - mx and my <= y <= self.height - my

    def to_px(self, lat: float, lon: float) -> tuple[float, float]:
        return (lon - self.lt_lon) / self.dlon, (lat - self.lt_lat) / self.dlat

    def to_geo(self, x: float, y: float) -> tuple[float, float]:
        return self.lt_lat + self.dlat * y, self.lt_lon + self.dlon * x

    def read_px(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        for (r, c), part in self.parts.items():
            px0, py0 = int(self.col_x[c]), int(self.row_y[r])
            ax0, ay0 = max(x0, px0), max(y0, py0)
            ax1, ay1 = min(x1, px0 + part.width), min(y1, py0 + part.height)
            if ax0 < ax1 and ay0 < ay1:
                out[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] = part.read(
                    ax0 - px0, ay0 - py0, ax1 - px0, ay1 - py0
                )
        return out

    def crop(self, lat: float, lon: float, size_m: float, out_px: int = 512) -> Image.Image:
        """North-up square of ``size_m`` ground metres centred on (lat, lon)."""
        cx, cy = self.to_px(lat, lon)
        w, h = size_m / self.gsd_x, size_m / self.gsd_y
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        x1, y1 = x0 + max(8, int(round(w))), y0 + max(8, int(round(h)))
        return Image.fromarray(self.read_px(x0, y0, x1, y1)).resize((out_px, out_px), Image.BILINEAR)
