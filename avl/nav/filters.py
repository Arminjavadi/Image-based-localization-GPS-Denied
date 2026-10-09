"""Fusion-filter registry and the run loop that drives it over an IMU stream.

Only the error-state EKF is implemented (design doc §11); the registry and the
``FusionFilter`` protocol are the extension point for complementary / UKF /
particle filters later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from avl.nav.eskf import ErrorStateEKF, EskfConfig, UpdateInfo
from avl.nav.ins import NominalState


class FusionFilter(Protocol):
    state: NominalState

    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None: ...

    def update(self, z_enu: np.ndarray, confidence: float, spread_m: float) -> UpdateInfo: ...

    def pos_sigma_h(self) -> float: ...


FILTERS: dict[str, type] = {"eskf": ErrorStateEKF}


@dataclass
class Fix:
    """One VPR position measurement, expressed in the local ENU frame."""

    frame: int                 # index into the trajectory frame list
    grid_idx: int              # index into the IMU stream time grid
    pos_enu: np.ndarray        # (2,) or (3,)
    confidence: float
    spread_m: float


@dataclass
class FusionResult:
    t: np.ndarray                      # (N,) IMU grid
    pos: np.ndarray                    # (N, 3) fused position, ENU
    vel: np.ndarray                    # (N, 3)
    quat: np.ndarray                   # (N, 4)
    pos_sigma_h: np.ndarray            # (N,) filter's own horizontal 1-sigma
    bias_acc: np.ndarray              # (N, 3)
    bias_gyro: np.ndarray             # (N, 3)
    fix_frames: list[int] = field(default_factory=list)
    fix_gated: list[bool] = field(default_factory=list)
    fix_nees: list[float] = field(default_factory=list)

    def at_frames(self, frame_grid_idx: np.ndarray) -> np.ndarray:
        return self.pos[np.asarray(frame_grid_idx, dtype=int)]


def run_filter(
    imu,
    fixes: list[Fix],
    x0: NominalState,
    P0: np.ndarray,
    config: EskfConfig | None = None,
    name: str = "eskf",
    resolve: Callable[[Fix, FusionFilter], Fix | None] | None = None,
) -> FusionResult:
    """Propagate ``x0`` over ``imu`` and apply each fix at its grid index.

    ``resolve``, when given, is called with each fix and the filter just before
    the fix is applied, and returns the fix to use (or ``None`` to skip it). This
    is how a measurement can depend on the filter's own prediction, e.g. VPR
    searching only a window around the predicted position.
    """
    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}; have {sorted(FILTERS)}")
    filt: FusionFilter = FILTERS[name](x0, P0, config)

    n = len(imu)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    sig = np.zeros(n)
    ba = np.zeros((n, 3))
    bg = np.zeros((n, 3))
    by_grid: dict[int, Fix] = {int(f.grid_idx): f for f in fixes}

    res = FusionResult(imu.t.copy(), pos, vel, quat, sig, ba, bg)

    def _record(i: int) -> None:
        pos[i] = filt.state.pos
        vel[i] = filt.state.vel
        quat[i] = filt.state.quat
        sig[i] = filt.pos_sigma_h()
        ba[i] = filt.state.bias_acc
        bg[i] = filt.state.bias_gyro

    def _apply(f: Fix) -> None:
        if resolve is not None:
            f = resolve(f, filt)
            if f is None:
                return
        info = filt.update(f.pos_enu, f.confidence, f.spread_m)
        res.fix_frames.append(f.frame)
        res.fix_gated.append(info.gated)
        res.fix_nees.append(info.nees)

    # apply a fix that lands exactly on t[0]
    if 0 in by_grid:
        _apply(by_grid[0])
    _record(0)

    for i in range(1, n):
        filt.predict(imu.gyro[i - 1], imu.accel[i - 1], imu.dt)
        if i in by_grid:
            _apply(by_grid[i])
        _record(i)

    return res
