"""Trajectory-mode localization: VPR fused with odometry.

Two families of filter (``--filter``):

* ``eskf`` - VPR + simulated IMU through the 15-state error-state EKF (default).
* ``kf | kf_reanchor | kf_window | pf | pgo | hybrid`` - frame-rate 2-D filters
  from :mod:`avl.nav.vo_fusion`, driven by *real* frame-to-frame visual odometry
  (:mod:`avl.nav.sparse_vo`) with an IMU coast where VO fails, and fusing the full
  per-tile similarity map (the particle filter uses all of it, not only the top-5).
  ``hybrid`` (particle filter supervising a windowed KF) is the recommended one,
  see docs/VO_AVL_Fusion_Experiment.md.

The time-ordered counterpart to scripts/visloc_eval.py. For every frame of a
UAV-VisLoc trajectory it runs the *same* retrieval + geo-fusion as the rest of the
pipeline (avl.retrieval.localize), then fuses that per-frame pose with a synthetic
IMU (avl.nav) so the estimate is continuous, outlier-gated, and survives fix
dropouts. Writes <tag>_traj.csv / <tag>_traj.json for the GUI's Trajectory tab.

See docs/VisualInertial_AVL_Design.md (esp. sections 3-9, 12).

Example
-------
python scripts/visloc_traj.py \
    --refs data/visloc_avl/r10_t200/references.csv \
    --queries data/visloc_avl/r10_t200/queries.csv \
    --region-csv data/UAVVisLoc/10/10.csv \
    --model denseuav-vit --imu-grade consumer --vpr-every 1 \
    --ref-cache artifacts/visloc/cache_r10_t200__denseuav-vit.npy --tag r10_traj
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from avl.geo import DEFAULT_FUSION_METHOD, FUSION_METHODS, search_window
from avl.nav import (
    EskfConfig,
    Fix,
    GradePreset,
    LocalFrame,
    NominalState,
    PRESETS,
    default_p0,
    mechanize,
    run_filter,
    simulate_imu,
    trajectory_metrics,
)
from avl.nav.metrics import along_cross_track, horizontal_error, path_length
from avl.nav.vo_fusion import FRAME_FILTERS

# avl.retrieval pulls in torch/faiss; only imported for the real-VPR path
NATIVE_DIM = {
    "denseuav-vit", "game4loc", "sample4geo", "anyloc-lite", "anyloc-l", "anyloc-g",
    "dinov2", "dinov2-ft", "mixvpr", "cosplace", "eigenplaces",
}

THRESHOLDS_M = [25, 50, 100, 200, 500]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--refs", type=Path, default=None,
                   help="reference CSV (real mode); also used to locate the region mosaic")
    p.add_argument("--queries", type=Path, default=None,
                   help="time-ordered trajectory query CSV (real mode)")
    p.add_argument("--model", default="denseuav-vit", choices=sorted(NATIVE_DIM))
    p.add_argument("--rotations", type=int, default=4, choices=[1, 4])
    p.add_argument("--query-crop", default="square", choices=["none", "square"])
    p.add_argument("--north-align", action="store_true",
                   help="turn each frame north-up from its heading column (full-frame rotate, "
                        "then the largest square with no padding); pair with --rotations 1")
    p.add_argument("--yaw-sign", type=float, default=-1.0, choices=[1.0, -1.0],
                   help="rotate by sign*heading, counter-clockwise (-1 = clockwise by heading)")
    p.add_argument("--query-scales", type=float, nargs="+", default=[1.0],
                   help="centre-crop fractions searched alongside the orientations")
    p.add_argument("--fusion", default=DEFAULT_FUSION_METHOD, choices=list(FUSION_METHODS),
                   help="geo-fusion for the per-frame VPR pose")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)

    p.add_argument("--frame-stride", type=int, default=1, help="use every Nth query row")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--frame-subset", type=Path, default=None,
                   help="text file: one image_id per line, ordered (Trajectory Studio)")

    p.add_argument("--region-csv", type=Path, default=None,
                   help="UAV-VisLoc region CSV to source real frame timestamps (joined on id/stem)")
    p.add_argument("--frame-dt", type=float, default=3.0,
                   help="uniform frame spacing (s) when no timestamp is available")

    # synthetic-mission mode: the drawn route IS the flight path, VPR fixes are
    # GT + a magnitude sampled from a real benchmark's error distribution
    p.add_argument("--synthetic-path", type=Path, default=None,
                   help="JSON {waypoints:[[lat,lon],...], speed_mps, alt_m, fix_dt_s}: "
                        "route becomes the GT path; no retrieval is run")
    p.add_argument("--vpr-error-csv", type=Path, default=None,
                   help="visloc_eval <tag>_per_query.csv; synthetic fix error is sampled "
                        "from its fused_error_m / confidence / spread_m columns")
    p.add_argument("--vpr-sigma", type=float, default=None,
                   help="synthetic-mode fallback when no --vpr-error-csv: 1-sigma fix error (m)")
    p.add_argument("--vpr-outlier-rate", type=float, default=0.1,
                   help="fraction of synthetic fixes made gross outliers (--vpr-sigma path)")

    p.add_argument("--imu-rate", type=float, default=100.0)
    p.add_argument("--imu-grade", default="consumer", choices=[*PRESETS, "perfect"])
    p.add_argument("--imu-scale", type=float, default=1.0,
                   help="multiply every IMU error magnitude (quick 'custom' knob)")
    p.add_argument("--filter", default="eskf", choices=["eskf", *FRAME_FILTERS],
                   help="eskf = IMU-driven error-state EKF; the others fuse real visual odometry "
                        "with the per-tile similarity map (needs real retrieval)")
    p.add_argument("--camera-k", type=float, default=None,
                   help="VO ground scale: full-frame ground width / AGL (default: per-region constant)")
    p.add_argument("--vo-cache", type=Path, default=None,
                   help="optional .npz to reuse/store the VO steps of this frame list")
    p.add_argument("--dem-dir", type=Path, default=Path("data/dem/copernicus_glo30"))
    p.add_argument("--fuse-altitude", action="store_true")
    p.add_argument("--init-vel", choices=["known", "zero"], default="known")

    p.add_argument("--prior-k", type=float, default=None,
                   help="real mode: VPR searches only tiles within k x the filter's predicted "
                        "horizontal 1-sigma of its predicted position (off = global search)")
    p.add_argument("--prior-min-radius", type=float, default=0.0,
                   help="lower bound (m) on the search-window radius")
    p.add_argument("--prior-escape", type=float, default=None,
                   help="re-acquisition: if the best tile outside the window out-scores the best "
                        "inside by more than this (cosine), use the global result instead and let "
                        "the chi-square gate judge it. Guards against an overconfident filter "
                        "windowing out the truth.")
    p.add_argument("--scores-cache", type=Path, default=None,
                   help="real mode: .npy of per-frame tile scores to reuse across runs that share "
                        "model, map, frames and crop/rotations (for window/escape sweeps)")
    p.add_argument("--vpr-every", type=int, default=1, help="feed a fix to the filter every Nth used frame")
    p.add_argument("--dropout", default="", help="mission-time windows with no fix, e.g. '120-180,300-330'")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--tag", default="traj")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/visloc"))
    p.add_argument("--ref-cache", type=Path, default=None)
    p.add_argument("--vocab", type=Path, default=None)
    p.add_argument("--head-weights", type=Path, default=None,
                   help="dinov2-ft: trained head checkpoint (default artifacts/finetune/head.pt)")
    return p.parse_args()


def _dropouts(spec: str) -> list[tuple[float, float]]:
    out = []
    for chunk in filter(None, (c.strip() for c in spec.split(","))):
        a, b = chunk.split("-")
        out.append((float(a), float(b)))
    return out


def _first_col(frame: pd.DataFrame, *names: str) -> str | None:
    for n in names:
        if n in frame.columns:
            return n
    return None


def _resolve_paths(paths: pd.Series, csv_path: Path) -> list[Path]:
    base = csv_path.parent
    return [Path(p) if Path(p).is_absolute() else (base / p) for p in paths.astype(str)]


# --------------------------------------------------------------------------- #
# trajectory assembly
# --------------------------------------------------------------------------- #
def load_trajectory(args: argparse.Namespace) -> pd.DataFrame:
    q = pd.read_csv(args.queries)
    missing = {"image_path", "latitude", "longitude"} - set(q.columns)
    if missing:
        raise SystemExit(f"{args.queries} missing column(s): {', '.join(sorted(missing))}")

    id_col = _first_col(q, "image_id")
    q["_id"] = q[id_col].astype(str) if id_col else [f"row_{i}" for i in range(len(q))]
    q["_path"] = [str(p) for p in _resolve_paths(q["image_path"], args.queries)]
    q["_stem"] = q["_path"].map(lambda s: Path(s).stem)

    alt_col = _first_col(q, "altitude_m", "height_m", "height")
    q["_alt"] = q[alt_col].astype(float) if alt_col else np.nan
    yaw_col = _first_col(q, "yaw_deg", "Phi1", "heading_deg")
    q["_yaw"] = q[yaw_col].astype(float) if yaw_col else np.nan

    # frame subset / stride
    if args.frame_subset is not None:
        wanted = [ln.strip() for ln in args.frame_subset.read_text().splitlines() if ln.strip()]
        by_id = {r["_id"]: r for _, r in q.iterrows()}
        rows = [by_id[w] for w in wanted if w in by_id]
        if not rows:
            raise SystemExit("--frame-subset matched no query ids")
        q = pd.DataFrame(rows).reset_index(drop=True)
    else:
        q = q.iloc[:: args.frame_stride].reset_index(drop=True)
        if args.max_frames:
            q = q.iloc[: args.max_frames].reset_index(drop=True)

    # timestamps: prefer real per-frame times (they carry the true, irregular
    # spacing — uniform spacing turns a multi-second gap into an acceleration
    # spike in the IMU sim). Fall back to uniform only when real times are
    # unavailable or non-monotonic (a Studio "path order" subset re-orders frames).
    t_col = _first_col(q, "t_s", "time_s", "timestamp", "date")
    times: np.ndarray | None = None
    if args.region_csv is not None:
        reg = pd.read_csv(args.region_csv)
        t0 = datetime.fromisoformat(str(reg["date"].iloc[0]))
        stem_map = {
            Path(str(fn)).stem: (datetime.fromisoformat(str(d)) - t0).total_seconds()
            for fn, d in zip(reg["filename"], reg["date"])
        }
        cand = np.array([stem_map.get(s, np.nan) for s in q["_stem"]])
        if np.isnan(cand).any():
            print("[traj] --region-csv did not cover every frame; using uniform spacing", flush=True)
        else:
            times = cand
    elif t_col == "date":
        base = datetime.fromisoformat(str(q["date"].iloc[0]))
        times = np.array(
            [(datetime.fromisoformat(str(d)) - base).total_seconds() for d in q["date"]]
        )
    elif t_col:
        times = q[t_col].astype(float).to_numpy()

    if times is None:
        times = args.frame_dt * np.arange(len(q), dtype=np.float64)
    elif np.any(np.diff(times) <= 0):
        print("[traj] recorded timestamps not monotonic for this frame order; "
              "using uniform spacing", flush=True)
        times = args.frame_dt * np.arange(len(q), dtype=np.float64)

    q["_t"] = times - times[0]
    return q


# --------------------------------------------------------------------------- #
# synthetic-mission mode
# --------------------------------------------------------------------------- #
def _box_smooth(v: np.ndarray, win: int) -> np.ndarray:
    """Odd-window moving average with edge-reflect padding (endpoints stay put)."""
    win = max(1, int(win) | 1)
    if win <= 1 or v.size < 3:
        return v
    pad = win // 2
    padded = np.concatenate([v[pad:0:-1], v, v[-2 : -pad - 2 : -1]])
    kern = np.ones(win) / win
    return np.convolve(padded, kern, mode="valid")[: v.size]


def _round_corners(xy: np.ndarray, min_radius_m: float, fine_step: float = 5.0) -> np.ndarray:
    """Resample a polyline finely, then smooth so no corner is sharper than
    ``min_radius_m`` — a hand-drawn kink otherwise becomes an acceleration spike
    when the IMU sim differentiates the path twice."""
    seg = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    sf = np.arange(0.0, total + 1e-6, fine_step)
    fx = np.interp(sf, cum, xy[:, 0])
    fy = np.interp(sf, cum, xy[:, 1])
    win = int(round(min_radius_m / fine_step))
    for _ in range(2):
        fx = _box_smooth(fx, win)
        fy = _box_smooth(fy, win)
    return np.stack([fx, fy], axis=1)


def synth_mission(args):
    """Turn a drawn polyline into a GT flight path + benchmark-sampled VPR fixes."""
    spec = json.loads(args.synthetic_path.read_text())
    wp = np.asarray(spec["waypoints"], dtype=np.float64)
    speed = float(spec.get("speed_mps", 20.0))
    alt = float(spec.get("alt_m", 0.0))
    fix_dt = float(spec.get("fix_dt_s", 3.0))
    max_lat_accel = float(spec.get("max_lat_accel", 4.0))  # m/s^2, sets the turn radius
    if wp.ndim != 2 or wp.shape[0] < 2:
        raise SystemExit("synthetic path needs >= 2 [lat, lon] waypoints")
    have_csv = args.vpr_error_csv is not None and args.vpr_error_csv.is_file()
    if not have_csv and args.vpr_sigma is None:
        raise SystemExit("synthetic mode needs --vpr-error-csv or --vpr-sigma")

    lf = LocalFrame(lat0=float(wp[0, 0]), lon0=float(wp[0, 1]), alt0=alt)
    wpen = lf.geo_to_enu(wp[:, 0], wp[:, 1])[:, :2]
    if float(np.hypot(*np.diff(wpen, axis=0).T).sum()) < 1.0:
        raise SystemExit("synthetic path is shorter than 1 m")

    # a drone cannot fly a mathematical corner: round the drawn kinks to the
    # radius implied by a gentle lateral acceleration, then resample at frame rate
    r_min = max(40.0, speed * speed / max(max_lat_accel, 0.5))
    curve = _round_corners(wpen, r_min)
    ccum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])
    ctot = float(ccum[-1])
    step = max(speed * fix_dt, 1.0)
    s = np.arange(0.0, ctot + 1e-6, step)
    if s.size < 4:
        s = np.linspace(0.0, ctot, 4)
    gt_en = np.stack([np.interp(s, ccum, curve[:, 0]), np.interp(s, ccum, curve[:, 1])], axis=1)
    lat, lon, _ = lf.enu_to_geo(gt_en[:, 0], gt_en[:, 1], np.zeros(s.size))
    n = s.size
    heading = np.gradient(gt_en, axis=0)
    yaw = np.degrees(np.arctan2(heading[:, 0], heading[:, 1]))
    t_s = fix_dt * np.arange(n)

    rng = np.random.default_rng(args.seed + 101)
    if have_csv:
        ecsv = pd.read_csv(args.vpr_error_csv)
        err_pool = pd.to_numeric(ecsv.get("fused_error_m"), errors="coerce").dropna().to_numpy()
        if err_pool.size == 0:
            raise SystemExit(f"{args.vpr_error_csv} has no usable 'fused_error_m' column")
        conf_pool = pd.to_numeric(ecsv.get("confidence"), errors="coerce").dropna().to_numpy()
        spread_pool = pd.to_numeric(ecsv.get("spread_m"), errors="coerce").dropna().to_numpy()
        mag = rng.choice(err_pool, size=n)
        conf = rng.choice(conf_pool, size=n) if conf_pool.size else np.full(n, 0.6)
        spread = rng.choice(spread_pool, size=n) if spread_pool.size else np.full(n, 120.0)
        src = f"~ {args.vpr_error_csv.name} (median {np.median(err_pool):.0f} m)"
        floor = float(np.median(err_pool))
    else:
        sig = float(args.vpr_sigma)
        mag = np.abs(rng.normal(0.0, sig, size=n))
        n_out = int(round(args.vpr_outlier_rate * n))
        if n_out:
            mag[rng.choice(n, size=n_out, replace=False)] = rng.uniform(3 * sig, 12 * sig, size=n_out)
        conf = np.full(n, 0.6)
        spread = np.full(n, 2.0 * sig)
        src = f"synthetic ±{sig:.0f} m, {args.vpr_outlier_rate * 100:.0f}% outliers"
        floor = sig
    ang = rng.uniform(0.0, 2 * np.pi, size=n)
    fix_en = gt_en + np.stack([mag * np.cos(ang), mag * np.sin(ang)], axis=1)
    flat, flon, _ = lf.enu_to_geo(fix_en[:, 0], fix_en[:, 1], np.zeros(n))
    print(f"[traj] synthetic mission · {n} frames · {ctot / 1000:.2f} km · {speed:.0f} m/s · "
          f"turn r>={r_min:.0f} m · VPR error {src}", flush=True)
    return {
        "n": n, "t_s": t_s, "gt_lat": lat, "gt_lon": lon, "gt_alt": np.full(n, alt),
        "yaw": yaw, "has_yaw": True, "alt0": alt, "query_ids": [f"wp_{i:04d}" for i in range(n)],
        "vpr_lat": flat, "vpr_lon": flon, "vpr_alt": np.full(n, np.nan),
        "vpr_conf": conf, "vpr_spread": spread, "vpr_err": mag, "query_s": 0.0,
        # match the filter's measurement noise to the sampled error level so a
        # single noisy fix cannot yank position/velocity (calibration §8 assumes a
        # tighter, better-behaved encoder)
        "sigma_floor_m": float(np.clip(floor, 20.0, 400.0)),
    }


def real_mission(args):
    """Run real retrieval on the trajectory query frames (imports torch/faiss)."""
    from PIL import Image

    from avl.retrieval import (
        build_encoder,
        encode_reference_map,
        load_reference_map,
        localize,
        query_scores,
    )

    if args.refs is None or args.queries is None:
        raise SystemExit("real mode needs --refs and --queries (or use --synthetic-path)")
    traj = load_trajectory(args)
    n = len(traj)
    gt_lat = traj["latitude"].to_numpy(dtype=np.float64)
    gt_lon = traj["longitude"].to_numpy(dtype=np.float64)
    gt_alt = traj["_alt"].to_numpy(dtype=np.float64)
    print(f"[traj] {n} frames  {traj['_t'].iloc[-1]:.0f}s  model {args.model}  "
          f"fusion {args.fusion}  imu {args.imu_grade}x{args.imu_scale:g}", flush=True)

    references = load_reference_map(args.refs)
    t0 = time.perf_counter()
    encoder = build_encoder(args.model, references, vocab_path=args.vocab,
                            batch_size=args.batch_size, rotations=args.rotations,
                            head_weights=args.head_weights)
    print(f"[traj] {args.model} ready on {encoder.device} in {time.perf_counter() - t0:.1f}s", flush=True)
    ref_desc, cached = encode_reference_map(encoder, references, args.ref_cache, args.batch_size)
    print(f"[traj] {len(references)} ref tiles, descriptors {ref_desc.shape} "
          f"{'(cache)' if cached else ''}", flush=True)

    vpr_lat = np.full(n, np.nan)
    vpr_lon = np.full(n, np.nan)
    vpr_alt = np.full(n, np.nan)
    vpr_conf = np.zeros(n)
    vpr_spread = np.zeros(n)
    vpr_err = np.full(n, np.nan)
    # Per-tile scores are kept so the windowed search (--prior-k) can re-select
    # inside the filter loop without encoding the frame a second time.
    frame_scores: list[np.ndarray] = []

    headings = args.yaw_sign * traj["_yaw"].to_numpy(dtype=np.float64) if args.north_align else None

    def frame_heading(i: int) -> float | None:
        return None if headings is None or not np.isfinite(headings[i]) else float(headings[i])

    def localize_frame(i: int, window: np.ndarray | None = None):
        return localize(
            encoder, Path(traj["_path"].iloc[i]), references, ref_desc,
            rotations=args.rotations, query_crop=args.query_crop, top_k=args.top_k,
            ground_truth=(gt_lat[i], gt_lon[i]), batch_size=args.batch_size,
            fusion_method=args.fusion, window=window, scores=frame_scores[i],
            scales=tuple(args.query_scales), heading_deg=frame_heading(i),
        )

    cached = None
    if args.scores_cache is not None and args.scores_cache.exists():
        cached = np.load(args.scores_cache)
        if cached.shape != (n, len(references)):
            print(f"[traj] scores cache {cached.shape} does not fit; recomputing", flush=True)
            cached = None
    qt0 = time.perf_counter()
    for i in range(n):
        if cached is not None:
            frame_scores.append(cached[i])
        else:
            with Image.open(traj["_path"].iloc[i]) as handle:
                image = handle.convert("RGB")
            frame_scores.append(query_scores(
                encoder, image, ref_desc, args.rotations, args.query_crop, args.batch_size,
                tuple(args.query_scales), frame_heading(i),
            ))
        res = localize_frame(i)
        vpr_lat[i] = res.fused.latitude
        vpr_lon[i] = res.fused.longitude
        vpr_alt[i] = res.fused.altitude_m if res.fused.altitude_m is not None else np.nan
        vpr_conf[i] = res.confidence
        vpr_spread[i] = res.spread_m
        vpr_err[i] = res.fused_error_m if res.fused_error_m is not None else np.nan
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"[traj]   {i + 1}/{n} frames localized", flush=True)
    if args.scores_cache is not None and cached is None:
        args.scores_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.scores_cache, np.stack(frame_scores))
    gt_alt0 = float(gt_alt[0]) if np.isfinite(gt_alt[0]) else 0.0
    yaw = traj["_yaw"].to_numpy(dtype=np.float64)
    return {
        "n": n, "t_s": traj["_t"].to_numpy(dtype=np.float64), "n_refs": len(references),
        "gt_lat": gt_lat, "gt_lon": gt_lon, "gt_alt": gt_alt,
        "yaw": yaw, "has_yaw": not np.isnan(yaw).any(), "alt0": gt_alt0,
        "query_ids": traj["_id"].to_numpy(),
        "vpr_lat": vpr_lat, "vpr_lon": vpr_lon, "vpr_alt": vpr_alt,
        "vpr_conf": vpr_conf, "vpr_spread": vpr_spread, "vpr_err": vpr_err,
        "query_s": time.perf_counter() - qt0,
        "localize_frame": localize_frame,
        "frame_scores": frame_scores,
        "paths": traj["_path"].tolist(),
        "stems": traj["_stem"].tolist(),
        "ref_lat": references.latitude, "ref_lon": references.longitude,
    }


# --------------------------------------------------------------------------- #
# frame-rate filters: real visual odometry + the per-tile similarity map
# --------------------------------------------------------------------------- #
def _attitude(region_csv: Path | None, stems: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """UAV-VisLoc camera tilt (Omega, Kappa) per frame, or (None, None)."""
    if region_csv is None:
        return None, None
    reg = pd.read_csv(region_csv)
    if not {"Omega", "Kappa", "filename"} <= set(reg.columns):
        return None, None
    reg.index = reg["filename"].map(lambda f: Path(str(f)).stem)
    att = reg.reindex(stems)
    if att[["Omega", "Kappa"]].isna().any().any():
        return None, None
    return att["Omega"].to_numpy(dtype=float), att["Kappa"].to_numpy(dtype=float)


def frame_filter_mission(args, fr: dict, lf: LocalFrame, gt_enu: np.ndarray, t_s: np.ndarray,
                         has_fix: np.ndarray) -> dict:
    from avl.nav import vo_fusion as vf
    from avl.nav.sparse_vo import VoTrack, measure_track
    from avl.pipeline import camera_k_for, region_of
    from avl.terrain import TerrainModel

    if "frame_scores" not in fr:
        raise SystemExit(f"--filter {args.filter} needs real retrieval (not --synthetic-path)")
    n = fr["n"]
    k = args.camera_k
    if k is None:
        k, why = camera_k_for(region_of(args.refs))
        if k is None:
            raise SystemExit(f"visual odometry needs --camera-k ({why})")
    agl = TerrainModel(args.dem_dir).agl(fr["gt_alt"], fr["gt_lat"], fr["gt_lon"])
    omega, kappa = _attitude(args.region_csv, fr["stems"])

    track = None
    if args.vo_cache is not None and args.vo_cache.exists():
        z = np.load(args.vo_cache, allow_pickle=False)
        if list(z["stems"]) == list(fr["stems"]) and float(z["k"]) == float(k):
            track = VoTrack(z["steps"], z["ok"], z["inliers"], [""] * (n - 1), float(z["seconds"]))
            print(f"[traj] VO steps from {args.vo_cache}", flush=True)
    if track is None:
        print(f"[traj] visual odometry over {n} frames (k {k:.3f}, "
              f"{'tilt-corrected' if omega is not None else 'nadir assumed'})", flush=True)

        def progress(done: int, total: int) -> None:
            if done % 10 == 0 or done == total:
                print(f"[traj]   {done}/{total} frames odometry", flush=True)

        track = measure_track(fr["paths"], fr["yaw"], agl, k, omega, kappa, progress=progress)
        if args.vo_cache is not None:
            args.vo_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.vo_cache, steps=track.steps, ok=track.ok, inliers=track.inliers,
                     stems=np.array(fr["stems"]), k=k, seconds=track.seconds)

    rng = np.random.default_rng(args.seed)
    vo, sig, ok = vf.real_vo(gt_enu[:, :2], t_s, vf.RealVo(track.steps, track.ok), rng)
    refs_xy = lf.geo_to_enu(np.asarray(fr["ref_lat"]), np.asarray(fr["ref_lon"]))[:, :2]
    case = vf.Case(gt_enu[:, :2], refs_xy, np.stack(fr["frame_scores"]), vo, sig,
                   vf.tile_stride(refs_xy), has_fix)
    t0 = time.perf_counter()
    est = vf.METHODS[args.filter](case, np.random.default_rng(args.seed))
    filt_ms = 1000.0 * (time.perf_counter() - t0) / n
    return {
        "est_en": est, "vo_en": vf.vo_only(case), "vo_ok": ok, "camera_k": float(k),
        "vo_ms_per_frame": 1000.0 * track.seconds / n, "filter_ms_per_frame": filt_ms,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.prior_k is not None and args.synthetic_path:
        raise SystemExit("--prior-k needs real retrieval; it does not apply to --synthetic-path")

    fr = synth_mission(args) if args.synthetic_path else real_mission(args)
    n = fr["n"]
    t_s = fr["t_s"]
    gt_lat, gt_lon, gt_alt = fr["gt_lat"], fr["gt_lon"], fr["gt_alt"]
    yaw, has_yaw, alt0 = fr["yaw"], fr["has_yaw"], fr["alt0"]
    query_ids = fr["query_ids"]
    vpr_lat, vpr_lon, vpr_alt = fr["vpr_lat"], fr["vpr_lon"], fr["vpr_alt"]
    vpr_conf, vpr_spread, vpr_err = fr["vpr_conf"], fr["vpr_spread"], fr["vpr_err"]
    query_s = fr["query_s"]

    # ---- local frame + ground truth ENU --------------------------
    lf = LocalFrame(lat0=float(gt_lat[0]), lon0=float(gt_lon[0]), alt0=alt0)
    gt_enu = lf.geo_to_enu(gt_lat, gt_lon, np.where(np.isfinite(gt_alt), gt_alt, alt0))

    # ---- synthetic IMU + INS-only -------------------------------
    base_preset = GradePreset.perfect() if args.imu_grade == "perfect" else PRESETS[args.imu_grade]
    preset = base_preset if args.imu_scale == 1.0 else base_preset.scaled(args.imu_scale)
    imu = simulate_imu(t_s, gt_enu, yaw_deg=yaw if has_yaw else None,
                       rate_hz=args.imu_rate, preset=preset, seed=args.seed)
    v0 = imu.true_vel[0].copy() if args.init_vel == "known" else np.zeros(3)
    x0 = NominalState(pos=gt_enu[0].copy(), vel=v0, quat=imu.true_quat[0].copy())
    ins = mechanize(imu, x0)
    ins_enu = ins.pos[imu.frame_grid_idx]

    # ---- fuse -------------------------------------------------
    windows = _dropouts(args.dropout)
    m = 3 if args.fuse_altitude else 2
    fixes: list[Fix] = []
    fix_used = np.zeros(n, dtype=bool)
    for i in range(n):
        if i == 0 or i % args.vpr_every != 0:
            continue
        if any(a <= t_s[i] <= b for a, b in windows):
            continue
        alt_i = vpr_alt[i] if np.isfinite(vpr_alt[i]) else alt0
        p_enu = lf.geo_to_enu(vpr_lat[i], vpr_lon[i], alt_i)[:m]
        fixes.append(Fix(frame=i, grid_idx=int(imu.frame_grid_idx[i]),
                         pos_enu=p_enu, confidence=float(vpr_conf[i]),
                         spread_m=float(vpr_spread[i])))
        fix_used[i] = True

    # ---- windowed VPR: retrieval searches around the filter's own prediction ----
    windowed = args.prior_k is not None
    win_lat = np.full(n, np.nan)
    win_lon = np.full(n, np.nan)
    win_err = np.full(n, np.nan)
    win_tiles = np.zeros(n, dtype=int)
    win_radius = np.full(n, np.nan)
    win_escaped = np.zeros(n, dtype=bool)

    def resolve(f: Fix, filt) -> Fix:
        P_h = filt.P[0:2, 0:2]
        sigma = float(np.sqrt(np.linalg.eigvalsh(P_h).max()))
        radius = max(args.prior_k * sigma, args.prior_min_radius)
        c_lat, c_lon, _ = lf.enu_to_geo(filt.state.pos[0], filt.state.pos[1])
        window = search_window(fr["ref_lat"], fr["ref_lon"], float(c_lat), float(c_lon),
                               radius, min_keep=args.top_k)
        i = f.frame
        sc = fr["frame_scores"][i]
        escaped = (
            args.prior_escape is not None
            and (~window).any()
            and float(sc[~window].max()) - float(sc[window].max()) > args.prior_escape
        )
        res = fr["localize_frame"](i, None if escaped else window)
        win_escaped[i] = escaped
        win_lat[i], win_lon[i] = res.fused.latitude, res.fused.longitude
        win_err[i] = res.fused_error_m if res.fused_error_m is not None else np.nan
        win_tiles[i] = int(window.sum())
        win_radius[i] = radius
        alt_i = res.fused.altitude_m if res.fused.altitude_m is not None else alt0
        return Fix(frame=i, grid_idx=f.grid_idx,
                   pos_enu=lf.geo_to_enu(win_lat[i], win_lon[i], alt_i)[:m],
                   confidence=float(res.confidence), spread_m=float(res.spread_m))

    vo_res = None
    if args.filter == "eskf":
        cfg = EskfConfig(fuse_altitude=args.fuse_altitude)
        if "sigma_floor_m" in fr:  # synthetic mode: use the sampled error level, homoscedastic
            cfg.sigma_floor_m = fr["sigma_floor_m"]
            cfg.alpha_spread = 0.0
            cfg.beta_conf = 0.0
            print(f"[traj] filter R floor set to {cfg.sigma_floor_m:.0f} m (from the error pool)", flush=True)
        fused = run_filter(imu, fixes, x0, default_p0(), cfg, args.filter,
                           resolve=resolve if windowed else None)
        fused_enu = fused.at_frames(imu.frame_grid_idx)
        gated_frames = {fr for fr, g in zip(fused.fix_frames, fused.fix_gated) if g}
        P_sigma = fused.pos_sigma_h[imu.frame_grid_idx]
        ba_norm = np.linalg.norm(fused.bias_acc[imu.frame_grid_idx], axis=1)
        bg_norm = np.linalg.norm(fused.bias_gyro[imu.frame_grid_idx], axis=1)
        vel_f = fused.vel[imu.frame_grid_idx]
        n_acc = int(sum(not g for g in fused.fix_gated))
        n_gate = int(sum(fused.fix_gated))
        nees = fused.fix_nees
    else:
        if windowed:
            print("[traj] --prior-k applies to eskf only (kf_window / hybrid search their own window)",
                  flush=True)
        vo_res = frame_filter_mission(args, fr, lf, gt_enu, t_s, fix_used)
        fused_enu = np.c_[vo_res["est_en"], gt_enu[:, 2]]      # altitude: barometer
        gated_frames = set()
        P_sigma = np.full(n, np.nan)
        ba_norm = bg_norm = np.full(n, np.nan)
        vel_f = np.full((n, 3), np.nan)
        n_acc, n_gate, nees = int(fix_used.sum()), 0, None

    # ---- back to geo + errors ---------------------------------
    ins_lat, ins_lon, ins_alt = lf.enu_to_geo(ins_enu[:, 0], ins_enu[:, 1], ins_enu[:, 2])
    fu_lat, fu_lon, fu_alt = lf.enu_to_geo(fused_enu[:, 0], fused_enu[:, 1], fused_enu[:, 2])
    err_ins = horizontal_error(ins_enu, gt_enu)
    err_fused = horizontal_error(fused_enu, gt_enu)
    along, cross = along_cross_track(fused_enu, gt_enu)

    # ---- metrics --------------------------------------------
    path_m = path_length(gt_enu)
    have_vpr = np.isfinite(vpr_err)

    ins_metrics = trajectory_metrics(ins_enu, gt_enu, total_path_m=path_m, n_frames=n)
    fused_metrics = trajectory_metrics(
        fused_enu, gt_enu, total_path_m=path_m, n_frames=n,
        n_fix_accepted=n_acc, n_fix_gated=n_gate, nees=nees,
        fuse_altitude=args.fuse_altitude,
    )
    vpr_vals = vpr_err[have_vpr]
    vpr_metrics = {
        "horiz_error_m": {
            "mean": float(vpr_vals.mean()), "median": float(np.median(vpr_vals)),
            "p95": float(np.percentile(vpr_vals, 95)), "min": float(vpr_vals.min()),
            "max": float(vpr_vals.max()),
        },
        "cep50_m": float(np.percentile(vpr_vals, 50)),
        "cep95_m": float(np.percentile(vpr_vals, 95)),
        "rmse_m": float(np.sqrt(np.mean(vpr_vals**2))),
        "n_frames_with_fix": int(have_vpr.sum()),
        "n_outliers_gt_200m": int((vpr_vals > 200).sum()),
        "within_m": {str(k): float((vpr_vals <= k).mean()) for k in THRESHOLDS_M},
    }
    fused_metrics["within_m"] = {str(k): float((err_fused <= k).mean()) for k in THRESHOLDS_M}
    vo_metrics = None
    if vo_res is not None:
        vo_enu = np.c_[vo_res["vo_en"], gt_enu[:, 2]]
        vo_metrics = trajectory_metrics(vo_enu, gt_enu, total_path_m=path_m, n_frames=n)
        vo_metrics["steps_ok"] = float(np.mean(vo_res["vo_ok"]))
        vo_metrics["camera_k"] = vo_res["camera_k"]
        vo_metrics["ms_per_frame"] = vo_res["vo_ms_per_frame"]
        fused_metrics["filter_ms_per_frame"] = vo_res["filter_ms_per_frame"]
    win_metrics = None
    if windowed:
        wv = win_err[np.isfinite(win_err)]
        win_metrics = {
            "prior_k": args.prior_k,
            "prior_min_radius_m": args.prior_min_radius,
            "horiz_error_m": {
                "mean": float(wv.mean()), "median": float(np.median(wv)),
                "p95": float(np.percentile(wv, 95)), "min": float(wv.min()),
                "max": float(wv.max()),
            },
            "n_fixes": int(wv.size),
            "n_outliers_gt_200m": int((wv > 200).sum()),
            "window_tiles_mean": float(win_tiles[fix_used].mean()),
            "window_radius_m_median": float(np.nanmedian(win_radius)),
            "prior_escape": args.prior_escape,
            "n_escaped": int(win_escaped.sum()),
            "within_m": {str(k): float((wv <= k).mean()) for k in THRESHOLDS_M},
        }

    # ---- write CSV ---------------------------------------
    df = pd.DataFrame({
        "frame": np.arange(n),
        "t_s": t_s,
        "query_id": np.asarray(query_ids),
        "gt_lat": gt_lat, "gt_lon": gt_lon, "gt_alt": gt_alt,
        "ins_lat": ins_lat, "ins_lon": ins_lon, "ins_alt": ins_alt,
        "vpr_lat": vpr_lat, "vpr_lon": vpr_lon,
        "vpr_conf": vpr_conf, "vpr_spread_m": vpr_spread,
        "fused_lat": fu_lat, "fused_lon": fu_lon, "fused_alt": fu_alt,
        "err_ins_m": err_ins, "err_vpr_m": vpr_err, "err_fused_m": err_fused,
        "along_m": along, "cross_m": cross,
        "fix_used": fix_used.astype(int),
        "fix_gated": np.array([1 if i in gated_frames else 0 for i in range(n)]),
        "vpr_win_lat": win_lat, "vpr_win_lon": win_lon, "err_vpr_win_m": win_err,
        "win_tiles": win_tiles, "win_radius_m": win_radius, "win_escaped": win_escaped.astype(int),
        "P_pos_sigma_m": P_sigma,
        "vx": vel_f[:, 0], "vy": vel_f[:, 1], "vz": vel_f[:, 2],
        "bias_a_norm": ba_norm, "bias_g_norm": bg_norm,
    })
    csv_path = args.out_dir / f"{args.tag}_traj.csv"
    df.to_csv(csv_path, index=False)

    # ---- region mosaic (best effort) ------------------------
    mosaic = _region_mosaic(args.refs)

    payload = {
        "tag": args.tag,
        "mode": "synthetic" if args.synthetic_path else "real",
        "model": args.model,
        "fusion": args.fusion,
        "filter": args.filter,
        "odometry": "imu" if args.filter == "eskf" else "visual odometry (+ IMU coast)",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "refs_csv": str(args.refs),
        "queries_csv": str(args.queries),
        "imu": {
            "grade": args.imu_grade, "scale": args.imu_scale,
            "rate_hz": args.imu_rate, "seed": args.seed,
        },
        "vpr_every": args.vpr_every,
        "frame_stride": args.frame_stride,
        "fuse_altitude": bool(args.fuse_altitude),
        "init_vel": args.init_vel,
        "dropouts_s": windows,
        "n_frames": n,
        "n_refs": int(fr.get("n_refs", 0)),
        "duration_s": float(t_s[-1]),
        "path_length_m": float(path_m),
        "query_ms_per_frame": 1000.0 * query_s / max(1, n),
        "origin": {"lat": float(gt_lat[0]), "lon": float(gt_lon[0]), "alt": alt0},
        "region_mosaic": mosaic,
        "ins_only": ins_metrics,
        "vpr_only": vpr_metrics,
        "vpr_windowed": win_metrics,
        "vo_only": vo_metrics,
        "fused": fused_metrics,
        "series": {
            "t_s": t_s.round(3).tolist(),
            "gt_en": gt_enu[:, :2].round(2).tolist(),
            "ins_en": ins_enu[:, :2].round(2).tolist(),
            "vpr_en": [
                (lf.geo_to_enu(vpr_lat[i], vpr_lon[i])[:2].round(2).tolist()
                 if np.isfinite(vpr_lat[i]) else None)
                for i in range(n)
            ],
            "fused_en": fused_enu[:, :2].round(2).tolist(),
            "err_ins_m": err_ins.round(2).tolist(),
            "err_vpr_m": [None if not np.isfinite(v) else round(float(v), 2) for v in vpr_err],
            "err_fused_m": err_fused.round(2).tolist(),
            "P_sigma_m": [None if not np.isfinite(v) else round(float(v), 2) for v in P_sigma],
            "vo_en": vo_res["vo_en"].round(2).tolist() if vo_res is not None else None,
            "fix_used": fix_used.astype(int).tolist(),
            "fix_gated": [1 if i in gated_frames else 0 for i in range(n)],
        },
    }
    json_path = args.out_dir / f"{args.tag}_traj.json"
    json_path.write_text(json.dumps(payload, indent=2))

    # ---- console summary --------------------------------
    def line(name: str, mtr: dict, extra: str = "") -> None:
        v = mtr["horiz_error_m"]
        print(f"  {name:11s} median {v['median']:8.1f}  p95 {v['p95']:8.1f}  "
              f"max {v['max']:9.1f}  final {mtr.get('final_error_m', float('nan')):8.1f}  {extra}")

    src = "synthetic mission" if args.synthetic_path else f"{fr.get('n_refs', 0)} tiles"
    print(f"\n[traj] {n} frames · {src} · path {path_m/1000:.2f} km · "
          f"{payload['query_ms_per_frame']:.0f} ms/frame")
    print("horizontal error vs ground truth:")
    line("IMU-only", ins_metrics, f"drift {ins_metrics['drift_rate_pct']:.2f}%")
    if vo_metrics:
        line("VO-only", vo_metrics, f"drift {vo_metrics['drift_rate_pct']:.2f}%  "
             f"steps ok {100 * vo_metrics['steps_ok']:.0f}%  {vo_metrics['ms_per_frame']:.0f} ms/frame")
    line("VPR-only", vpr_metrics, f"({vpr_metrics['n_frames_with_fix']} fixes, "
         f"{vpr_metrics['n_outliers_gt_200m']} >200m)")
    if win_metrics:
        line("VPR-window", win_metrics, f"({win_metrics['n_fixes']} fixes, "
             f"{win_metrics['n_outliers_gt_200m']} >200m, {win_metrics['window_tiles_mean']:.1f} tiles, "
             f"r~{win_metrics['window_radius_m_median']:.0f}m, {win_metrics['n_escaped']} escaped)")
    line(f"Fused/{args.filter}", fused_metrics, f"acc {n_acc} gated {n_gate} "
         f"avail {fused_metrics.get('fix_availability', 0):.2f}")
    print(f"  fused within: " + "  ".join(
        f"{k}m {100*fused_metrics['within_m'][str(k)]:5.1f}%" for k in THRESHOLDS_M))
    print(f"  wrote {json_path}  and  {csv_path}\n")
    print("RESULT_JSON " + json.dumps({
        "tag": args.tag, "json": str(json_path), "csv": str(csv_path),
        "fused_median_m": fused_metrics["horiz_error_m"]["median"],
        "fused_p95_m": fused_metrics["horiz_error_m"]["p95"],
        "vpr_median_m": vpr_metrics["horiz_error_m"]["median"],
        "vpr_window_median_m": win_metrics["horiz_error_m"]["median"] if win_metrics else None,
        "ins_drift_pct": ins_metrics["drift_rate_pct"],
    }), flush=True)
    return 0


def _region_mosaic(refs_csv: Path) -> dict | None:
    """Best-effort: map a visloc_avl refs path (…/rNN_tMMM/…) to its mosaic corners."""
    import re

    match = re.search(r"r(\d+)_t\d+", str(refs_csv))
    if not match:
        return None
    region = match.group(1)
    ranges = Path("data/UAVVisLoc/satellite_ coordinates_range.csv")
    if not ranges.is_file():
        return None
    try:
        rf = pd.read_csv(ranges)
        rf["_rid"] = rf["mapname"].str.extract(r"satellite(\d+)")
        row = rf[rf["_rid"] == region]
        if row.empty:
            return None
        row = row.iloc[0]
        return {
            "path": f"data/UAVVisLoc/{region}/{row['mapname']}",
            "lt": [float(row["LT_lat_map"]), float(row["LT_lon_map"])],
            "rb": [float(row["RB_lat_map"]), float(row["RB_lon_map"])],
        }
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
