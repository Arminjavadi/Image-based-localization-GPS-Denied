"""Cache frozen DINOv2 token features for training / evaluating a retrieval head.

Two modes:

region  a UAV-VisLoc region used for *training*: every Nth drone frame at the four
        search orientations, a north-up satellite crop centred on its ground truth
        at each of several ground scales (the positives), and random crops of the
        same mosaic (extra negatives).
csv     a prepared reference/query set (scripts/visloc_prepare.py layout), used for
        validation and test: every reference tile, and the queries at 4 orientations.

Example
-------
python scripts/finetune_features.py region --region 02 --frame-stride 2
python scripts/finetune_features.py csv --refs data/visloc_avl/r10_t200/references.csv \
    --queries data/visloc_avl/r10_t200/queries.csv --max-queries 144 --name r10_t200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from avl.finetune.backbone import (
    TRANSFORM,
    load_backbone,
    open_query,
    query_rotations,
)
from avl.mosaic import RegionMosaic

OUT_DIR = Path("artifacts/finetune/feats")


def encode(model, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
    out = []
    for s in range(0, len(images), batch_size):
        batch = torch.stack([TRANSFORM(im) for im in images[s : s + batch_size]])
        out.append(model(batch).numpy().astype(np.float16))
    return np.concatenate(out)


def resolve(paths: pd.Series, csv_path: Path) -> list[str]:
    base = csv_path.parent
    return [p if Path(p).is_absolute() else str(base / p) for p in paths.astype(str)]


def progress(tag: str, i: int, n: int, t0: float) -> None:
    rate = (time.perf_counter() - t0) / max(1, i)
    print(f"[{tag}] {i}/{n}  {rate:.1f} s/item  eta {rate * (n - i) / 60:.0f} min", flush=True)


def run_region(args, model) -> None:
    root = Path(args.root)
    region = args.region
    mosaic = RegionMosaic(root, region)
    frames = pd.read_csv(root / region / f"{region}.csv").iloc[:: args.frame_stride]
    margin = max(args.scales) / 2
    keep = [
        (root / region / "drone" / fn).exists() and mosaic.contains(lat, lon, margin)
        for fn, lat, lon in zip(frames["filename"], frames["lat"], frames["lon"])
    ]
    frames = frames[keep].reset_index(drop=True)
    n = len(frames)
    tag = f"r{region}"
    print(f"[{tag}] {n} frames, mosaic {mosaic.width}x{mosaic.height} px, "
          f"GSD {mosaic.gsd_x:.2f}x{mosaic.gsd_y:.2f} m, scales {args.scales}", flush=True)
    if n == 0:  # e.g. region 07, whose map is a 50 m strip
        raise SystemExit(f"[{tag}] no frame has a full-size crop inside the map; skipping region")

    S = len(args.scales)
    q_feat = np.zeros((n, 4, 65, 768), np.float16)
    t_feat = np.zeros((n, S, 65, 768), np.float16)
    t0 = time.perf_counter()
    for i, row in frames.iterrows():
        images = query_rotations(open_query(str(root / region / "drone" / row["filename"])))
        images += [mosaic.crop(row["lat"], row["lon"], s) for s in args.scales]
        f = encode(model, images)
        q_feat[i], t_feat[i] = f[:4], f[4:]
        if (i + 1) % 25 == 0 or i + 1 == n:
            progress(tag, i + 1, n, t0)

    # random gallery crops anywhere on the map (rejecting mostly-padding ones)
    rng = np.random.default_rng(int(region))
    g_lat, g_lon, g_scale, g_img = [], [], [], []
    tries = 0
    while len(g_img) < args.gallery and tries < args.gallery * 20:
        tries += 1
        s = float(rng.choice(args.scales))
        mx, my = s / 2 / mosaic.gsd_x, s / 2 / mosaic.gsd_y
        lat, lon = mosaic.to_geo(rng.uniform(mx, mosaic.width - mx), rng.uniform(my, mosaic.height - my))
        im = mosaic.crop(lat, lon, s)
        if (np.asarray(im.resize((32, 32))).max(axis=2) < 8).mean() > 0.2:
            continue
        g_lat.append(lat), g_lon.append(lon), g_scale.append(s), g_img.append(im)
    g_feat = encode(model, g_img) if g_img else np.zeros((0, 65, 768), np.float16)
    print(f"[{tag}] {len(g_img)} gallery crops", flush=True)

    out = OUT_DIR / f"train_r{region}.npz"
    np.savez(
        out,
        q_feat=q_feat, t_feat=t_feat, scales=np.asarray(args.scales, np.float32),
        lat=frames["lat"].to_numpy(np.float64), lon=frames["lon"].to_numpy(np.float64),
        ids=frames["filename"].astype(str).to_numpy(),
        g_feat=g_feat, g_lat=np.asarray(g_lat), g_lon=np.asarray(g_lon),
        g_scale=np.asarray(g_scale, np.float32),
    )
    print(f"[{tag}] wrote {out} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def run_csv(args, model) -> None:
    refs = pd.read_csv(args.refs)
    queries = pd.read_csv(args.queries).iloc[:: args.query_stride]
    if args.max_queries:
        queries = queries.iloc[: args.max_queries]
    tag = args.name
    t0 = time.perf_counter()
    ref_paths = resolve(refs["image_path"], args.refs)
    r_feat = []
    for s in range(0, len(ref_paths), 64):
        r_feat.append(encode(model, [Image.open(p).convert("RGB") for p in ref_paths[s : s + 64]]))
        progress(f"{tag} refs", min(s + 64, len(ref_paths)), len(ref_paths), t0)
    r_feat = np.concatenate(r_feat)

    q_paths = resolve(queries["image_path"], args.queries)
    q_feat = np.zeros((len(q_paths), 4, 65, 768), np.float16)
    t0 = time.perf_counter()
    for i, p in enumerate(q_paths):
        q_feat[i] = encode(model, query_rotations(open_query(p)))
        if (i + 1) % 25 == 0 or i + 1 == len(q_paths):
            progress(f"{tag} queries", i + 1, len(q_paths), t0)

    out = OUT_DIR / f"eval_{tag}.npz"
    np.savez(
        out,
        r_feat=r_feat, r_lat=refs["latitude"].to_numpy(np.float64),
        r_lon=refs["longitude"].to_numpy(np.float64), r_ids=refs["image_id"].astype(str).to_numpy(),
        q_feat=q_feat, q_lat=queries["latitude"].to_numpy(np.float64),
        q_lon=queries["longitude"].to_numpy(np.float64),
        q_ids=queries["image_id"].astype(str).to_numpy(),
    )
    print(f"[{tag}] wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("region")
    r.add_argument("--root", default="data/UAVVisLoc")
    r.add_argument("--region", required=True)
    r.add_argument("--frame-stride", type=int, default=2)
    r.add_argument("--scales", type=float, nargs="+", default=[175.0, 250.0, 350.0])
    r.add_argument("--gallery", type=int, default=300)
    c = sub.add_parser("csv")
    c.add_argument("--refs", type=Path, required=True)
    c.add_argument("--queries", type=Path, required=True)
    c.add_argument("--query-stride", type=int, default=1)
    c.add_argument("--max-queries", type=int, default=None)
    c.add_argument("--name", required=True)
    args = parser.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_backbone("cpu")
    (run_region if args.mode == "region" else run_csv)(args, model)


if __name__ == "__main__":
    main()
