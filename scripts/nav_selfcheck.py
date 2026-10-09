"""Phase-1 sanity check for avl.nav on a real UAV-VisLoc trajectory.

No encoder / torch: the VPR fixes are *simulated* as ground truth plus Gaussian
noise, so this only exercises the IMU synthesis, the strapdown INS, and the
error-state Kalman filter. It prints the three-way IMU-only / VPR-only / Fused
error comparison the trajectory GUI tab will show once retrieval is wired in
(scripts/visloc_traj.py, Phase 2).

Example
-------
python scripts/nav_selfcheck.py --region-csv data/UAVVisLoc/10/10.csv \
    --imu-grade consumer --vpr-noise-m 25 --dropout 180-240
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
from avl.nav.metrics import horizontal_error, path_length


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region-csv", type=Path, default=Path("data/UAVVisLoc/10/10.csv"),
                   help="UAV-VisLoc region CSV: num,filename,date,lat,lon,height,...,Phi1")
    p.add_argument("--imu-grade", default="consumer", choices=[*PRESETS, "perfect"])
    p.add_argument("--imu-rate", type=float, default=100.0)
    p.add_argument("--vpr-noise-m", type=float, default=25.0, help="1-sigma of the simulated VPR fix.")
    p.add_argument("--vpr-every", type=int, default=1, help="feed a fix every Nth frame")
    p.add_argument("--dropout", default="", help="mission-time windows with no fix, e.g. '120-180,300-330'")
    p.add_argument("--init-vel", choices=["known", "zero"], default="known")
    p.add_argument("--fuse-altitude", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _dropouts(spec: str) -> list[tuple[float, float]]:
    out = []
    for chunk in filter(None, (c.strip() for c in spec.split(","))):
        a, b = chunk.split("-")
        out.append((float(a), float(b)))
    return out


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.region_csv)
    t0 = datetime.fromisoformat(str(df["date"].iloc[0]))
    t_frames = np.array([(datetime.fromisoformat(str(d)) - t0).total_seconds() for d in df["date"]])
    lat = df["lat"].to_numpy(float)
    lon = df["lon"].to_numpy(float)
    alt = df["height"].to_numpy(float)
    yaw = df["Phi1"].to_numpy(float) if "Phi1" in df else np.zeros(len(df))

    lf = LocalFrame(lat0=float(lat[0]), lon0=float(lon[0]), alt0=float(alt[0]))
    gt_enu = lf.geo_to_enu(lat, lon, alt)

    preset = GradePreset.perfect() if args.imu_grade == "perfect" else PRESETS[args.imu_grade]
    imu = simulate_imu(t_frames, gt_enu, yaw_deg=yaw, rate_hz=args.imu_rate, preset=preset, seed=args.seed)

    v0 = imu.true_vel[0].copy() if args.init_vel == "known" else np.zeros(3)
    x0 = NominalState(pos=gt_enu[0].copy(), vel=v0, quat=imu.true_quat[0].copy())

    # ---- IMU-only -----------------------------------------------------
    ins = mechanize(imu, x0)
    ins_frames = ins.pos[imu.frame_grid_idx]

    # ---- simulated VPR fixes ----------------------------------------
    rng = np.random.default_rng(args.seed + 1)
    windows = _dropouts(args.dropout)
    m = 3 if args.fuse_altitude else 2
    vpr_frames = np.full((len(df), 3), np.nan)
    fixes: list[Fix] = []
    for k, gi in enumerate(imu.frame_grid_idx):
        if k % args.vpr_every != 0:
            continue
        if any(a <= t_frames[k] <= b for a, b in windows):
            continue
        noise = rng.normal(0.0, args.vpr_noise_m, size=m)
        meas = gt_enu[k, :m] + noise
        vpr_frames[k, :m] = meas
        if k == 0:
            continue
        fixes.append(Fix(frame=k, grid_idx=int(gi), pos_enu=meas, confidence=0.75, spread_m=2 * args.vpr_noise_m))

    cfg = EskfConfig(fuse_altitude=args.fuse_altitude)
    res = run_filter(imu, fixes, x0, default_p0(), cfg, "eskf")
    fused_frames = res.at_frames(imu.frame_grid_idx)

    # ---- metrics ----------------------------------------------------
    path_m = path_length(gt_enu)
    have_vpr = ~np.isnan(vpr_frames[:, 0])
    n_acc = sum(not g for g in res.fix_gated)
    n_gate = sum(res.fix_gated)

    def show(name: str, est: np.ndarray, **extra) -> None:
        mtr = trajectory_metrics(est, gt_enu, total_path_m=path_m, n_frames=len(df), **extra)
        v = mtr["horiz_error_m"]
        print(f"  {name:10s} median {v['median']:7.1f}  p95 {v['p95']:7.1f}  "
              f"max {v['max']:8.1f}  final {mtr['final_error_m']:7.1f}  "
              f"drift {mtr['drift_rate_pct']:5.2f}%")

    print(f"\nregion CSV : {args.region_csv}")
    print(f"frames     : {len(df)}   duration {t_frames[-1]:.0f}s   path {path_m/1000:.2f} km   "
          f"mean speed {path_m/t_frames[-1]:.1f} m/s")
    print(f"IMU        : {preset.name}   rate {args.imu_rate:.0f} Hz   grid {len(imu)} samples")
    print(f"VPR sim    : sigma {args.vpr_noise_m:.0f} m   every {args.vpr_every} frame(s)   "
          f"dropout {windows or 'none'}   fixes accepted {n_acc}  gated {n_gate}")
    print("\nhorizontal error vs ground truth:")
    show("IMU-only", ins_frames)
    vpr_err = horizontal_error(np.nan_to_num(vpr_frames, nan=1e9), gt_enu)
    vpr_err = vpr_err[have_vpr]
    print(f"  {'VPR-only':10s} median {np.median(vpr_err):7.1f}  p95 {np.percentile(vpr_err,95):7.1f}  "
          f"max {vpr_err.max():8.1f}  (raw fixes, {have_vpr.sum()} of {len(df)} frames)")
    show("Fused", fused_frames, n_fix_accepted=n_acc, n_fix_gated=n_gate, nees=res.fix_nees,
         fuse_altitude=args.fuse_altitude)
    print()


if __name__ == "__main__":
    main()
