"""Trajectory-error metrics (see docs/VisualInertial_AVL_Design.md §9).

Everything is evaluated at the trajectory frame times, in the local ENU frame.
Horizontal error uses the east/north components only.
"""

from __future__ import annotations

import numpy as np


def _stats(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def horizontal_error(est_enu: np.ndarray, gt_enu: np.ndarray) -> np.ndarray:
    d = np.asarray(est_enu, dtype=np.float64)[:, :2] - np.asarray(gt_enu, dtype=np.float64)[:, :2]
    return np.hypot(d[:, 0], d[:, 1])


def along_cross_track(est_enu: np.ndarray, gt_enu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed along-track and cross-track error, using the GT heading at each step."""
    est = np.asarray(est_enu, dtype=np.float64)[:, :2]
    gt = np.asarray(gt_enu, dtype=np.float64)[:, :2]
    err = est - gt
    head = np.gradient(gt, axis=0)
    norm = np.hypot(head[:, 0], head[:, 1])
    norm[norm < 1e-9] = 1e-9
    that = head / norm[:, None]
    nhat = np.stack([-that[:, 1], that[:, 0]], axis=1)
    along = np.einsum("ij,ij->i", err, that)
    cross = np.einsum("ij,ij->i", err, nhat)
    return along, cross


def path_length(gt_enu: np.ndarray) -> float:
    p = np.asarray(gt_enu, dtype=np.float64)[:, :2]
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def trajectory_metrics(
    est_enu: np.ndarray,
    gt_enu: np.ndarray,
    *,
    total_path_m: float | None = None,
    n_frames: int | None = None,
    n_fix_accepted: int = 0,
    n_fix_gated: int = 0,
    nees: list[float] | None = None,
    fuse_altitude: bool = False,
) -> dict:
    est = np.asarray(est_enu, dtype=np.float64)
    gt = np.asarray(gt_enu, dtype=np.float64)
    e_h = horizontal_error(est, gt)
    along, cross = along_cross_track(est, gt)
    path = float(total_path_m) if total_path_m is not None else path_length(gt)

    out: dict = {
        "horiz_error_m": _stats(e_h),
        "along_track_m": _stats(np.abs(along)),
        "cross_track_m": _stats(np.abs(cross)),
        "cep50_m": float(np.percentile(e_h, 50)),
        "cep95_m": float(np.percentile(e_h, 95)),
        "final_error_m": float(e_h[-1]),
        "rmse_m": float(np.sqrt(np.mean(e_h**2))),
        "path_length_m": path,
        "drift_rate_pct": float(100.0 * e_h[-1] / path) if path > 0 else float("nan"),
    }
    if fuse_altitude and est.shape[1] > 2:
        out["vert_error_m"] = _stats(np.abs(est[:, 2] - gt[:, 2]))
    if n_frames:
        out["fix_availability"] = float(n_fix_accepted / n_frames)
    out["fixes_accepted"] = int(n_fix_accepted)
    out["fixes_gated"] = int(n_fix_gated)
    if nees:
        out["mean_nees"] = float(np.mean(nees))
    return out
