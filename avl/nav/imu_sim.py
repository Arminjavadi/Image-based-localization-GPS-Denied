"""Synthetic IMU generator (see docs/VisualInertial_AVL_Design.md §5).

UAV-VisLoc ships no inertial data, so we reconstruct a C2-continuous trajectory
from the sparse ground-truth frames (cubic spline for position, interpolated
Euler angles for attitude), differentiate it to the ideal specific force and
angular rate, then corrupt those with a MEMS-grade error model:

    accel_meas = (I + S_a) f_b + b_a(t) + white_a
    gyro_meas  = (I + S_g) w_b + b_g(t) + white_g
    b(t) = b(0) + random walk

Grade presets (``PRESETS``) give order-of-magnitude figures for a consumer MEMS
unit and a tactical unit; ``GradePreset.perfect()`` is the noise-free case used
by the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from avl.nav.frames import (
    GRAVITY,
    gravity_enu,
    quat_conj,
    quat_from_euler,
    quat_mul,
    quat_normalize,
    quat_to_rot,
)


# --------------------------------------------------------------------------- #
# natural cubic spline (per scalar channel), no scipy
# --------------------------------------------------------------------------- #
class CubicSpline1D:
    """Natural cubic spline through ``(x, y)`` with analytic 1st/2nd derivatives."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("need at least two knots")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing")
        self.x = x
        self.y = y
        n = x.size
        h = np.diff(x)
        m = np.zeros(n, dtype=np.float64)  # second derivatives
        if n > 2:
            a = h[:-1].copy()
            b = 2.0 * (h[:-1] + h[1:])
            c = h[1:].copy()
            d = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
            # Thomas algorithm for the (n-2) tridiagonal system
            for i in range(1, n - 2):
                w = a[i] / b[i - 1]
                b[i] -= w * c[i - 1]
                d[i] -= w * d[i - 1]
            mid = np.zeros(n - 2)
            mid[-1] = d[-1] / b[-1]
            for i in range(n - 4, -1, -1):
                mid[i] = (d[i] - c[i] * mid[i + 1]) / b[i]
            m[1:-1] = mid
        self.m = m

    def __call__(self, t: np.ndarray, der: int = 0) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        idx = np.clip(np.searchsorted(self.x, t) - 1, 0, self.x.size - 2)
        x0 = self.x[idx]
        x1 = self.x[idx + 1]
        h = x1 - x0
        A = (x1 - t) / h
        B = (t - x0) / h
        y0 = self.y[idx]
        y1 = self.y[idx + 1]
        m0 = self.m[idx]
        m1 = self.m[idx + 1]
        if der == 0:
            out = A * y0 + B * y1 + ((A**3 - A) * m0 + (B**3 - B) * m1) * (h * h) / 6.0
        elif der == 1:
            out = (y1 - y0) / h - (3 * A**2 - 1) / 6.0 * h * m0 + (3 * B**2 - 1) / 6.0 * h * m1
        elif der == 2:
            out = A * m0 + B * m1
        else:
            raise ValueError("der must be 0, 1 or 2")
        return out


# --------------------------------------------------------------------------- #
# grade presets
# --------------------------------------------------------------------------- #
_DEG = np.pi / 180.0
_SQRT_HR = 60.0  # sqrt(3600 s)


@dataclass(frozen=True)
class GradePreset:
    """IMU error parameters, all in SI (rad, m, s).

    ``gyro_white`` / ``accel_white`` are noise densities (units * sqrt(s)); the
    per-sample white-noise std is ``density * sqrt(rate_hz)``. ``*_bias_rw`` are
    the random-walk driving densities (units/sqrt(s)); the per-step increment std
    is ``density * sqrt(dt)``.

    ``*_turnon`` is the **post-alignment residual** bias, not the raw datasheet
    zero-rate offset: trajectory mode starts from a known pose and a coarse
    alignment, so the uncalibrated turn-on offset (which for a raw consumer gyro
    can reach ~0.5 deg/s and would swamp everything) is assumed already removed.
    """

    name: str
    gyro_white: float          # rad/s * sqrt(s)
    gyro_bias_rw: float        # (rad/s) / sqrt(s)
    gyro_turnon: float         # rad/s, 1-sigma
    gyro_sf: float             # fraction, 1-sigma
    accel_white: float         # m/s^2 * sqrt(s)
    accel_bias_rw: float       # (m/s^2) / sqrt(s)
    accel_turnon: float        # m/s^2, 1-sigma
    accel_sf: float            # fraction, 1-sigma
    misalign: float            # rad, 1-sigma (off-diagonal of S)

    @classmethod
    def perfect(cls) -> "GradePreset":
        return cls("perfect", 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def scaled(self, factor: float) -> "GradePreset":
        """All error magnitudes multiplied by ``factor`` (1.0 = unchanged)."""
        f = float(factor)
        return replace(
            self,
            name=f"{self.name}x{f:g}",
            gyro_white=self.gyro_white * f,
            gyro_bias_rw=self.gyro_bias_rw * f,
            gyro_turnon=self.gyro_turnon * f,
            gyro_sf=self.gyro_sf * f,
            accel_white=self.accel_white * f,
            accel_bias_rw=self.accel_bias_rw * f,
            accel_turnon=self.accel_turnon * f,
            accel_sf=self.accel_sf * f,
            misalign=self.misalign * f,
        )


PRESETS: dict[str, GradePreset] = {
    # consumer MEMS  (e.g. ICM-42688 / BMI088 class)
    "consumer": GradePreset(
        name="consumer",
        gyro_white=0.30 * _DEG / _SQRT_HR,          # 0.30 deg/sqrt(hr)
        gyro_bias_rw=100.0 * _DEG / 3600.0 / _SQRT_HR,  # 100 deg/hr/sqrt(hr)
        gyro_turnon=0.02 * _DEG,                    # 0.02 deg/s residual after alignment
        gyro_sf=0.001,                              # 0.1 %
        accel_white=0.10 / _SQRT_HR,               # 0.10 (m/s)/sqrt(hr)
        accel_bias_rw=600e-6 * GRAVITY / _SQRT_HR,   # 600 ug/sqrt(hr)
        accel_turnon=3e-3 * GRAVITY,                # 3 mg residual after alignment
        accel_sf=0.002,                             # 0.2 %
        misalign=0.02 * _DEG,
    ),
    # tactical grade (e.g. STIM300 class)
    "tactical": GradePreset(
        name="tactical",
        gyro_white=0.05 * _DEG / _SQRT_HR,
        gyro_bias_rw=8.0 * _DEG / 3600.0 / _SQRT_HR,
        gyro_turnon=0.005 * _DEG,
        gyro_sf=0.0005,
        accel_white=0.02 / _SQRT_HR,
        accel_bias_rw=50e-6 * GRAVITY / _SQRT_HR,
        accel_turnon=0.5e-3 * GRAVITY,
        accel_sf=0.0005,
        misalign=0.01 * _DEG,
    ),
}


# --------------------------------------------------------------------------- #
# stream container
# --------------------------------------------------------------------------- #
@dataclass
class ImuStream:
    t: np.ndarray              # (N,) seconds, uniform
    dt: float
    gyro: np.ndarray           # (N, 3) rad/s, corrupted
    accel: np.ndarray          # (N, 3) m/s^2, corrupted (specific force)
    true_gyro: np.ndarray      # (N, 3) ideal
    true_accel: np.ndarray     # (N, 3) ideal specific force
    true_pos: np.ndarray       # (N, 3) ENU, from the spline
    true_vel: np.ndarray       # (N, 3) ENU
    true_quat: np.ndarray      # (N, 4) body->nav
    frame_grid_idx: np.ndarray  # (M,) index into t nearest each trajectory frame
    bias_acc: np.ndarray       # (N, 3) the true accel bias walk (for diagnostics)
    bias_gyro: np.ndarray      # (N, 3) the true gyro bias walk

    def __len__(self) -> int:
        return int(self.t.size)


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def simulate_imu(
    t_frames: np.ndarray,
    pos_enu: np.ndarray,
    *,
    yaw_deg: np.ndarray | None = None,
    roll_pitch_deg: np.ndarray | None = None,
    quats: np.ndarray | None = None,
    rate_hz: float = 100.0,
    preset: GradePreset | str = "consumer",
    seed: int | None = 0,
) -> ImuStream:
    """Synthesise an IMU stream consistent with a ground-truth trajectory.

    Parameters
    ----------
    t_frames : (M,) seconds, strictly increasing (from the frame timestamps).
    pos_enu  : (M, 3) ground-truth position in a local ENU frame, metres.
    yaw_deg  : (M,) heading, degrees. Roll/pitch default to level unless given.
    roll_pitch_deg : (M, 2) optional roll and pitch, degrees.
    quats    : (M, 4) optional body->nav quaternions; overrides the Euler inputs.
    rate_hz  : IMU sample rate.
    preset   : ``GradePreset`` or a key of ``PRESETS``.
    seed     : RNG seed for the (one-shot) bias / scale draw and the white noise.
    """
    if isinstance(preset, str):
        preset = PRESETS[preset]
    rng = np.random.default_rng(seed)

    t_frames = np.asarray(t_frames, dtype=np.float64)
    pos_enu = np.asarray(pos_enu, dtype=np.float64)
    m = t_frames.size
    if pos_enu.shape != (m, 3):
        raise ValueError("pos_enu must be (M, 3) matching t_frames")

    t0, t1 = float(t_frames[0]), float(t_frames[-1])
    dt = 1.0 / float(rate_hz)
    n = int(np.floor((t1 - t0) / dt)) + 1
    t = t0 + dt * np.arange(n)

    # ---- position spline -> pos / vel / accel -------------------------------
    sx = CubicSpline1D(t_frames, pos_enu[:, 0])
    sy = CubicSpline1D(t_frames, pos_enu[:, 1])
    sz = CubicSpline1D(t_frames, pos_enu[:, 2])
    true_pos = np.stack([sx(t), sy(t), sz(t)], axis=1)
    true_vel = np.stack([sx(t, 1), sy(t, 1), sz(t, 1)], axis=1)
    accel_nav = np.stack([sx(t, 2), sy(t, 2), sz(t, 2)], axis=1)

    # ---- attitude on the fine grid ---------------------------------------
    if quats is not None:
        quats = np.asarray(quats, dtype=np.float64)
        true_quat = _slerp_series(t_frames, quats, t)
    else:
        if yaw_deg is None:
            # heading from the velocity direction (ENU): yaw measured from north
            yaw = np.arctan2(true_vel[:, 0], true_vel[:, 1])
        else:
            yaw_u = np.unwrap(np.radians(np.asarray(yaw_deg, dtype=np.float64)))
            yaw = CubicSpline1D(t_frames, yaw_u)(t)
        if roll_pitch_deg is None:
            roll = np.zeros(n)
            pitch = np.zeros(n)
        else:
            rp = np.radians(np.asarray(roll_pitch_deg, dtype=np.float64))
            roll = CubicSpline1D(t_frames, rp[:, 0])(t)
            pitch = CubicSpline1D(t_frames, rp[:, 1])(t)
        true_quat = np.array([quat_from_euler(roll[i], pitch[i], yaw[i]) for i in range(n)])

    # ---- ideal sensor outputs ------------------------------------------
    g_n = gravity_enu()
    f_nav = accel_nav - g_n  # specific force in n
    R = np.array([quat_to_rot(q) for q in true_quat])  # (n, 3, 3)
    true_accel = np.einsum("nji,nj->ni", R, f_nav)  # R^T @ f_nav  (body frame)
    true_gyro = _angular_rate_from_quats(true_quat, dt)

    # ---- error model ---------------------------------------------------
    S_g = _scale_misalign(rng, preset.gyro_sf, preset.misalign)
    S_a = _scale_misalign(rng, preset.accel_sf, preset.misalign)
    bg0 = rng.normal(0.0, preset.gyro_turnon, size=3)
    ba0 = rng.normal(0.0, preset.accel_turnon, size=3)
    bg_walk = _random_walk(rng, n, preset.gyro_bias_rw, dt)
    ba_walk = _random_walk(rng, n, preset.accel_bias_rw, dt)
    bias_gyro = bg0 + bg_walk
    bias_acc = ba0 + ba_walk
    wn_g = rng.normal(0.0, preset.gyro_white * np.sqrt(rate_hz), size=(n, 3))
    wn_a = rng.normal(0.0, preset.accel_white * np.sqrt(rate_hz), size=(n, 3))

    gyro = true_gyro @ S_g.T + bias_gyro + wn_g
    accel = true_accel @ S_a.T + bias_acc + wn_a

    frame_grid_idx = np.clip(np.round((t_frames - t0) / dt).astype(int), 0, n - 1)

    return ImuStream(
        t=t,
        dt=dt,
        gyro=gyro,
        accel=accel,
        true_gyro=true_gyro,
        true_accel=true_accel,
        true_pos=true_pos,
        true_vel=true_vel,
        true_quat=true_quat,
        frame_grid_idx=frame_grid_idx,
        bias_acc=bias_acc,
        bias_gyro=bias_gyro,
    )


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _scale_misalign(rng, sf_std: float, mis_std: float) -> np.ndarray:
    S = np.eye(3)
    if sf_std > 0:
        S += np.diag(rng.normal(0.0, sf_std, size=3))
    if mis_std > 0:
        off = rng.normal(0.0, mis_std, size=(3, 3))
        np.fill_diagonal(off, 0.0)
        S += off
    return S


def _random_walk(rng, n: int, density: float, dt: float) -> np.ndarray:
    if density <= 0:
        return np.zeros((n, 3))
    steps = rng.normal(0.0, density * np.sqrt(dt), size=(n, 3))
    return np.cumsum(steps, axis=0)


def _angular_rate_from_quats(q: np.ndarray, dt: float) -> np.ndarray:
    """Body angular rate from a quaternion series via central differences.

    w_b = 2 * vec( conj(q) * qdot ).
    """
    n = q.shape[0]
    qdot = np.zeros_like(q)
    qdot[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    qdot[0] = (q[1] - q[0]) / dt
    qdot[-1] = (q[-1] - q[-2]) / dt
    w = np.zeros((n, 3))
    for i in range(n):
        omega = 2.0 * quat_mul(quat_conj(q[i]), qdot[i])
        w[i] = omega[1:]
    return w


def _slerp_series(t_knots: np.ndarray, q_knots: np.ndarray, t: np.ndarray) -> np.ndarray:
    q_knots = np.array([quat_normalize(qq) for qq in q_knots])
    idx = np.clip(np.searchsorted(t_knots, t) - 1, 0, t_knots.size - 2)
    out = np.zeros((t.size, 4))
    for k in range(t.size):
        i = idx[k]
        span = t_knots[i + 1] - t_knots[i]
        frac = 0.0 if span <= 0 else float((t[k] - t_knots[i]) / span)
        out[k] = _slerp_one(q_knots[i], q_knots[i + 1], np.clip(frac, 0.0, 1.0))
    return out


def _slerp_one(q0: np.ndarray, q1: np.ndarray, frac: float) -> np.ndarray:
    from avl.nav.frames import slerp

    return slerp(q0, q1, frac)
