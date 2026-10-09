#!/usr/bin/env python3
"""Real frame-to-frame visual odometry on UAV-VisLoc frames (see avl/nav/sparse_vo.py).

For each consecutive pair of frames in a map's ``queries.csv`` (recorded order):
SIFT + RANSAC similarity, corrected for camera tilt, scaled by height above ground
(baro - DEM) and the per-region camera constant k, rotated by the logged heading.
Writes one row per step with the VO estimate and the ground-truth step, so the VO
error can be measured and fed to ``scripts/vo_fusion_experiment.py --real-vo``.

    .venv/bin/python scripts/visloc_vo.py r05_t250 r06_t200 r10_t200 r11_t250
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avl.geo import _to_local_xy  # noqa: E402
from avl.nav.sparse_vo import SparseVO, nadir_pixel  # noqa: E402
from avl.pipeline import camera_k_for  # noqa: E402
from avl.rerank import load_gray  # noqa: E402
from avl.terrain import TerrainModel  # noqa: E402

DATA = Path("data/visloc_avl")
RAW = Path("data/UAVVisLoc")
OUT = Path("artifacts/visloc/vo_real")
MIN_AGL_M = 50.0      # take-off / landing frames: no usable ground scale


def run_map(job: tuple[str, int, int, float | None, bool]) -> Path:
    tag, max_frames, long_side, k_override, tilt = job
    region = tag.split("_")[0][1:]
    q = pd.read_csv(DATA / tag / "queries.csv")
    if max_frames:
        q = q.iloc[:max_frames]
    q = q.reset_index(drop=True)
    names = q.image_path.map(lambda p: Path(p).name)
    log = pd.read_csv(RAW / region / f"{region}.csv").set_index("filename").reindex(names)
    t = pd.to_datetime(log["date"].to_numpy())
    t = (t - t[0]).total_seconds().to_numpy(dtype=float)
    agl = TerrainModel().agl(q.height_m, q.latitude, q.longitude)
    k = k_override or camera_k_for(region)[0]
    x, y = _to_local_xy(q.latitude.to_numpy(), q.longitude.to_numpy(), q.latitude[0], q.longitude[0])
    gt = np.c_[x, y]

    vo = SparseVO(long_side=long_side)
    rows, prev = [], None
    t0 = time.time()
    for i, path in enumerate(q.image_path):
        gray = load_gray(path, long_side)
        size = (gray.shape[1], gray.shape[0])
        feat = vo.features(gray)
        focal = size[0] / k
        nad = nadir_pixel(size, focal, log.Omega.iloc[i], log.Kappa.iloc[i]) if tilt else None
        if prev is not None and min(agl[i - 1], agl[i]) < MIN_AGL_M:
            d = gt[i] - gt[i - 1]
            rows.append({"frame": i, "image_id": q.image_id.iloc[i], "dt_s": t[i] - t[i - 1], "ok": False,
                         "reason": f"below {MIN_AGL_M:.0f} m AGL", "gt_east_m": d[0], "gt_north_m": d[1],
                         "gt_step_m": float(np.hypot(*d))})
        elif prev is not None:
            feat_a, size_a, nad_a = prev
            step = vo.step(feat_a, feat, size_a, size, k * agl[i - 1] / size_a[0], q.yaw_deg.iloc[i - 1],
                           scale_hint=agl[i] / agl[i - 1], nadir_a=nad_a, nadir_b=nad)
            d = gt[i] - gt[i - 1]
            err = np.hypot(step.east_m - d[0], step.north_m - d[1])
            rows.append({"frame": i, "image_id": q.image_id.iloc[i], "dt_s": t[i] - t[i - 1],
                         "ok": step.ok, "reason": step.reason, "east_m": step.east_m, "north_m": step.north_m,
                         "inliers": step.inliers, "matches": step.matches, "scale": step.scale,
                         "rot_deg": step.rot_deg, "gt_east_m": d[0], "gt_north_m": d[1],
                         "gt_step_m": float(np.hypot(*d)), "err_m": err,
                         "rel_err": err / max(float(np.hypot(*d)), 1.0)})
        prev = (feat, size, nad)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{tag}{'' if tilt else '_notilt'}_vo.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    ok = df[df.ok]
    print(f"{tag}: {len(ok)}/{len(df)} steps ok, k={k:.3f}, median rel err {100 * ok.rel_err.median():.1f} %, "
          f"p90 {100 * ok.rel_err.quantile(0.9):.1f} %, median {ok.err_m.median():.1f} m "
          f"({(time.time() - t0) / len(q):.2f} s/frame)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps", nargs="+", help="map tags under data/visloc_avl, e.g. r05_t250")
    ap.add_argument("--max-frames", type=int, default=144)
    ap.add_argument("--long-side", type=int, default=1200)
    ap.add_argument("--camera-k", type=float, default=None, help="override the per-region k")
    ap.add_argument("--no-tilt", action="store_true", help="use the image centre, not the nadir pixel")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    jobs = [(m, args.max_frames, args.long_side, args.camera_k, not args.no_tilt) for m in args.maps]
    with Pool(min(args.jobs, len(jobs))) as pool:
        pool.map(run_map, jobs)


if __name__ == "__main__":
    main()
