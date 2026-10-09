"""Does the on-board localizer reproduce the benchmark?

Builds an AVLLocalizer index for a prepared map with a recipe preset, localizes the
first N query frames *in flight order* with their telemetry (heading, barometric
altitude), and compares every frame's top-1 tile with a scripts/visloc_eval.py run
of the same recipe. The two share avl.pipeline, so they should agree on every frame;
a few frames may differ by float ties. Exit code 1 when more than --tolerance differ.

The benchmark reads the terrain under the logged position when it simulates no
prior, so the localizer is given the logged position as its prior (without a
sigma, so it searches the whole map) to read the same terrain.

python scripts/check_localizer_parity.py --map data/visloc_avl/r05_t250 \\
    --camera-k 0.971 --eval-csv artifacts/visloc/scale/prod/r05_gated_per_query.csv \\
    --ref-cache artifacts/visloc/cache_r05_t250__megaloc.npy
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from avl.config import AVLConfig
from avl.geo import haversine_m
from avl.localizer import AVLLocalizer
from avl.pipeline import PRESETS, QueryTelemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, required=True, help="Prepared map folder (references.csv + queries.csv).")
    parser.add_argument("--preset", default="recommended", choices=sorted(PRESETS))
    parser.add_argument("--model", default=None, help="Override the preset's encoder.")
    parser.add_argument("--camera-k", type=float, default=None)
    parser.add_argument("--eval-csv", type=Path, required=True, help="visloc_eval *_per_query.csv to compare with.")
    parser.add_argument("--ref-cache", type=Path, default=None, help="Raw reference descriptors (.npy) to reuse.")
    parser.add_argument("--max-queries", type=int, default=144)
    parser.add_argument("--yaw-col", default="yaw_deg")
    parser.add_argument("--height-col", default="height_m")
    parser.add_argument("--tolerance", type=int, default=2, help="Frames allowed to disagree.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AVLConfig.from_preset(args.preset, device=args.device, score_threshold=-1.0)
    if args.model:
        config.model = args.model
    config.camera_k = args.camera_k

    refs_csv = args.map / "references.csv"
    queries = pd.read_csv(args.map / "queries.csv").iloc[: args.max_queries]
    expected = pd.read_csv(args.eval_csv)
    if len(expected) != len(queries):
        raise SystemExit(f"{args.eval_csv} has {len(expected)} frames, the map's first {len(queries)}")

    descriptors = np.load(args.ref_cache) if args.ref_cache else None
    localizer = AVLLocalizer(config)
    with tempfile.TemporaryDirectory() as index_dir:
        started = time.perf_counter()
        localizer.build_index(refs_csv, Path(index_dir), base_dir=args.map, descriptors=descriptors)
        print(f"[parity] index: {localizer.index.size} tiles, {config.model}, recipe heading="
              f"{config.heading_mode} center={config.center_mode} agl={config.scale_from_agl} "
              f"k={config.camera_k} tile={config.tile_m} m ({time.perf_counter() - started:.1f}s)",
              flush=True)

        rows = []
        for i, row in enumerate(queries.itertuples(index=False)):
            telemetry = QueryTelemetry(
                heading_deg=float(getattr(row, args.yaw_col)),
                altitude_asl_m=float(getattr(row, args.height_col)),
                prior_lat=float(row.latitude),
                prior_lon=float(row.longitude),
            )
            image = Path(row.image_path)
            if not image.is_absolute():
                image = args.map / image
            result = localizer.localize(image, telemetry=telemetry)
            top1 = result.best_match.record
            rows.append({
                "top1_id": str(top1.image_id),
                "top1_error_m": haversine_m(row.latitude, row.longitude, top1.latitude, top1.longitude),
                "crop_fraction": result.plan["crop_fraction"],
            })
            if (i + 1) % 24 == 0 or i + 1 == len(queries):
                print(f"[parity] {i + 1}/{len(queries)} frames", flush=True)

    got = pd.DataFrame(rows)
    same = got["top1_id"].to_numpy() == expected["top1_id"].astype(str).to_numpy()
    ours = 100 * float((got["top1_error_m"] <= 100).mean())
    theirs = 100 * float((expected["top1_error_m"] <= 100).mean())
    print(f"[parity] same top-1 tile on {int(same.sum())}/{len(same)} frames")
    print(f"[parity] top-1 within 100 m: localizer {ours:.1f}%  benchmark {theirs:.1f}%")
    if (~same).any():
        print(f"[parity] differing frames: {np.flatnonzero(~same).tolist()}")
    return 0 if int((~same).sum()) <= args.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())
