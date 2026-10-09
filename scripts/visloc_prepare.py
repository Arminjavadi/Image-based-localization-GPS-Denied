"""Build an AVL reference map and query set from one UAV-VisLoc region.

UAV-VisLoc ships one north-up georeferenced satellite mosaic per region plus a CSV
of geo-tagged drone frames. This script tiles the mosaic into square reference
tiles of a chosen *ground* size (metres), which is the knob that controls how well
the reference scale matches the drone frame footprint, and emits the reference /
query CSVs consumed by the rest of the pipeline.

Example
-------
python scripts/visloc_prepare.py --region 10 --tile-m 300 --overlap 0.5
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from avl.mosaic import RegionMosaic

Image.MAX_IMAGE_PIXELS = None

EARTH_A = 6_378_137.0


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """WGS-84 series for metres per degree of latitude and longitude."""
    lat = math.radians(lat_deg)
    m_lat = 111_132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_lon = 111_412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return m_lat, m_lon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/UAVVisLoc"))
    parser.add_argument("--region", required=True, help="Region id, e.g. 10")
    parser.add_argument(
        "--tile-m",
        type=float,
        required=True,
        help="Ground edge length of one reference tile, in metres.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Fractional overlap between neighbouring tiles (0.5 = 50%%).",
    )
    parser.add_argument(
        "--tile-px",
        type=int,
        default=512,
        help="Pixel size the tiles are resampled to (DenseUAV gallery uses 512).",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--offset-m", type=float, default=0.0,
                        help="Shift the tile grid's origin by this many metres (both axes); "
                             "a control for how much results depend on where tiles happen to fall.")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    region = args.region
    out_dir = args.out or Path("data/visloc_avl") / f"r{region}_t{int(args.tile_m)}"
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    ranges = pd.read_csv(args.root / "satellite_ coordinates_range.csv")
    ranges["region_id"] = ranges["mapname"].str.extract(r"satellite(\d+)")
    row = ranges[ranges["region_id"] == region]
    if row.empty:
        raise SystemExit(f"No satellite map entry for region {region}")
    row = row.iloc[0]

    # Windowed reads: the larger mosaics are several GB decoded, and region 09 is
    # split across four files (avl.mosaic handles both).
    try:
        mosaic = RegionMosaic(args.root, region)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing satellite map for region {region}: {exc}")

    lt_lat, lt_lon = float(row["LT_lat_map"]), float(row["LT_lon_map"])
    rb_lat, rb_lon = float(row["RB_lat_map"]), float(row["RB_lon_map"])

    width, height = mosaic.width, mosaic.height
    deg_per_px_lat = (rb_lat - lt_lat) / height  # negative: latitude decreases downwards
    deg_per_px_lon = (rb_lon - lt_lon) / width

    centre_lat = 0.5 * (lt_lat + rb_lat)
    m_lat, m_lon = meters_per_degree(centre_lat)
    gsd_y = abs(deg_per_px_lat) * m_lat
    gsd_x = abs(deg_per_px_lon) * m_lon

    tile_w = max(8, int(round(args.tile_m / gsd_x)))
    tile_h = max(8, int(round(args.tile_m / gsd_y)))
    stride_x = max(1, int(round(tile_w * (1.0 - args.overlap))))
    stride_y = max(1, int(round(tile_h * (1.0 - args.overlap))))

    print(f"region {region}: {row['region']}")
    print(f"  mosaic      {width} x {height} px, GSD {gsd_x:.3f} x {gsd_y:.3f} m/px")
    print(f"  ground      {width * gsd_x / 1000:.2f} x {height * gsd_y / 1000:.2f} km")
    print(f"  tile        {args.tile_m:.0f} m -> {tile_w} x {tile_h} px, stride {stride_x}/{stride_y}")

    rows = []
    n = 0
    off_x = int(round(args.offset_m / gsd_x))
    off_y = int(round(args.offset_m / gsd_y))
    for top in range(off_y, max(off_y + 1, height - tile_h + 1), stride_y):
        for left in range(off_x, max(off_x + 1, width - tile_w + 1), stride_x):
            pixels = mosaic.read_px(left, top, left + tile_w, top + tile_h)
            if not pixels.any():  # fully black padding region of the mosaic
                continue
            crop = Image.fromarray(pixels).resize((args.tile_px, args.tile_px), Image.BILINEAR)
            centre_px_x = left + tile_w / 2.0
            centre_px_y = top + tile_h / 2.0
            lat = lt_lat + deg_per_px_lat * centre_px_y
            lon = lt_lon + deg_per_px_lon * centre_px_x
            tile_id = f"r{region}_{n:06d}"
            rel = f"tiles/{tile_id}.jpg"
            crop.save(tiles_dir / f"{tile_id}.jpg", quality=args.jpeg_quality)
            rows.append(
                {
                    "image_path": rel,
                    "latitude": f"{lat:.8f}",
                    "longitude": f"{lon:.8f}",
                    "image_id": tile_id,
                }
            )
            n += 1
    print(f"  wrote       {n} reference tiles")

    ref_csv = out_dir / "references.csv"
    with ref_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "latitude", "longitude", "image_id"])
        writer.writeheader()
        writer.writerows(rows)

    queries = pd.read_csv(args.root / region / f"{region}.csv")
    drone_dir = (args.root / region / "drone").resolve()
    query_rows = []
    for _, q in queries.iterrows():
        path = drone_dir / str(q["filename"])
        if not path.exists():
            continue
        query_rows.append(
            {
                "image_path": str(path),
                "latitude": f"{float(q['lat']):.8f}",
                "longitude": f"{float(q['lon']):.8f}",
                "height_m": q.get("height", ""),
                "yaw_deg": q.get("Phi1", ""),
                "image_id": Path(str(q["filename"])).stem,
            }
        )
    query_csv = out_dir / "queries.csv"
    with query_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "latitude", "longitude", "height_m", "yaw_deg", "image_id"],
        )
        writer.writeheader()
        writer.writerows(query_rows)
    print(f"  wrote       {len(query_rows)} queries")
    print(f"  output      {out_dir}")


if __name__ == "__main__":
    main()
