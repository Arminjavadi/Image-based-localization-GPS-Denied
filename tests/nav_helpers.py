"""Shared fixtures for the avl.nav unit tests."""

from __future__ import annotations

import numpy as np


def analytic_traj(t: np.ndarray) -> np.ndarray:
    """A smooth curved ENU path, ~15 m/s, gently climbing. Returns (..., 3)."""
    t = np.asarray(t, dtype=np.float64)
    east = 15.0 * t + 20.0 * np.sin(0.05 * t)
    north = 40.0 * np.sin(0.03 * t)
    up = 100.0 + 2.0 * np.sin(0.02 * t)
    return np.stack([east, north, up], axis=-1)


def analytic_vel(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    de = 15.0 + 20.0 * 0.05 * np.cos(0.05 * t)
    dn = 40.0 * 0.03 * np.cos(0.03 * t)
    du = 2.0 * 0.02 * np.cos(0.02 * t)
    return np.stack([de, dn, du], axis=-1)


def frame_times(duration_s: float = 90.0, spacing_s: float = 3.0) -> np.ndarray:
    return np.arange(0.0, duration_s + 1e-9, spacing_s)


def yaw_along_velocity(t: np.ndarray) -> np.ndarray:
    """Heading (deg, from north, ENU) tangent to the analytic path."""
    v = analytic_vel(t)
    return np.degrees(np.arctan2(v[:, 0], v[:, 1]))
