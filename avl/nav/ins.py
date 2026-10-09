"""Strapdown inertial mechanization (see docs/VisualInertial_AVL_Design.md §6).

First-order integration of the nominal state ``x = (p, v, q, b_a, b_g)`` on the
IMU grid, in the local ENU navigation frame:

    q_{k+1} = q_k  (x)  dq((w_m - b_g) dt)
    a_n     = R{q_k} (f_m - b_a) + g_n
    v_{k+1} = v_k + a_n dt
    p_{k+1} = p_k + v_k dt + 1/2 a_n dt^2

Run with the filter's live bias estimates for the fused solution; run with zero
(or fixed turn-on) biases and no corrections for the "IMU-only" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from avl.nav.frames import (
    gravity_enu,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    quat_to_rot,
)


@dataclass
class NominalState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    bias_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "NominalState":
        return NominalState(
            self.pos.copy(),
            self.vel.copy(),
            quat_normalize(self.quat),
            self.bias_acc.copy(),
            self.bias_gyro.copy(),
        )


def mechanize_step(
    state: NominalState,
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: float,
    g_n: np.ndarray | None = None,
) -> NominalState:
    """One strapdown integration step. Biases pass through unchanged."""
    if g_n is None:
        g_n = gravity_enu()
    wb = np.asarray(gyro, dtype=np.float64) - state.bias_gyro
    fb = np.asarray(accel, dtype=np.float64) - state.bias_acc

    R = quat_to_rot(state.quat)
    a_n = R @ fb + g_n
    pos = state.pos + state.vel * dt + 0.5 * a_n * dt * dt
    vel = state.vel + a_n * dt
    quat = quat_normalize(quat_mul(state.quat, quat_from_rotvec(wb * dt)))
    return NominalState(pos, vel, quat, state.bias_acc.copy(), state.bias_gyro.copy())


@dataclass
class StateTrajectory:
    t: np.ndarray          # (N,)
    pos: np.ndarray        # (N, 3)
    vel: np.ndarray        # (N, 3)
    quat: np.ndarray       # (N, 4)

    def at_frames(self, frame_grid_idx: np.ndarray) -> np.ndarray:
        """Position rows at the given grid indices (the trajectory frame times)."""
        return self.pos[np.asarray(frame_grid_idx, dtype=int)]


def mechanize(imu, x0: NominalState) -> StateTrajectory:
    """Dead-reckon an entire ``ImuStream`` from ``x0`` with no corrections."""
    n = len(imu)
    g_n = gravity_enu()
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    st = x0.copy()
    pos[0], vel[0], quat[0] = st.pos, st.vel, st.quat
    for i in range(1, n):
        st = mechanize_step(st, imu.gyro[i - 1], imu.accel[i - 1], imu.dt, g_n)
        pos[i], vel[i], quat[i] = st.pos, st.vel, st.quat
    return StateTrajectory(t=imu.t.copy(), pos=pos, vel=vel, quat=quat)
