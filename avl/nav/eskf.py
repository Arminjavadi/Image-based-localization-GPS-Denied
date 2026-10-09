"""Error-state EKF for loosely-coupled visual-inertial fusion.

See docs/VisualInertial_AVL_Design.md §7-8. 15-element error state

    dx = [ dp(3)  dv(3)  dtheta(3)  db_a(3)  db_g(3) ]

with a **local** (right-multiplied) attitude error ``q = q_hat (x) dq(dtheta)``.
The nominal state is carried by the strapdown mechanization; the filter tracks
the error covariance, corrects the nominal state on each accepted VPR position
fix, and feeds the estimated biases back into the mechanization.

VPR fixes are gated by a chi-square test on the innovation, which is what lets
the filter reject the stray retrieval matches that a plain geo-fusion average
cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avl.nav.frames import gravity_enu, quat_from_rotvec, quat_mul, quat_normalize, quat_to_rot
from avl.nav.ins import NominalState, mechanize_step

P_ = slice(0, 3)
V_ = slice(3, 6)
TH_ = slice(6, 9)
BA_ = slice(9, 12)
BG_ = slice(12, 15)

# chi-square inverse-CDF table (dof 1..3), avoids a scipy dependency
_CHI2_PPF = {
    0.95: {1: 3.841, 2: 5.991, 3: 7.815},
    0.99: {1: 6.635, 2: 9.210, 3: 11.345},
    0.997: {1: 8.807, 2: 11.618, 3: 13.931},
    0.999: {1: 10.828, 2: 13.816, 3: 16.266},
}


def chi2_ppf(prob: float, dof: int) -> float:
    try:
        return _CHI2_PPF[round(prob, 3)][dof]
    except KeyError as exc:  # pragma: no cover - guardrail
        raise ValueError(
            f"no chi2 table entry for prob={prob}, dof={dof}; "
            f"available: {sorted(_CHI2_PPF)} x dof 1..3"
        ) from exc


@dataclass
class EskfConfig:
    # process noise (filter's model of the IMU; often inflated vs the true sim)
    accel_noise: float = 0.10 / 60.0          # (m/s^2) * sqrt(s)
    gyro_noise: float = 0.30 * np.pi / 180.0 / 60.0  # (rad/s) * sqrt(s)
    accel_bias_rw: float = 1000e-6 * 9.80665 / 60.0  # (m/s^2)/sqrt(s)
    gyro_bias_rw: float = 200.0 * np.pi / 180.0 / 3600.0 / 60.0  # (rad/s)/sqrt(s)
    # VPR measurement-noise mapping: sigma_h = floor + a*spread_m + b*(1-conf).
    # Defaults calibrated against anyloc-lite / denseuav-vit residuals on
    # UAV-VisLoc r05/r10 (docs/VisualInertial_AVL_Design.md §8): within-inlier
    # error is close to homoscedastic there, so the floor carries most of the
    # weight and the chi-square gate does the outlier rejection.
    sigma_floor_m: float = 45.0
    alpha_spread: float = 0.05
    beta_conf: float = 35.0
    gamma_z: float = 2.0
    # innovation gate
    gate_prob: float = 0.99
    fuse_altitude: bool = False

    def r_matrix(self, confidence: float, spread_m: float) -> np.ndarray:
        sig_h = self.sigma_floor_m + self.alpha_spread * float(spread_m) + self.beta_conf * (
            1.0 - float(confidence)
        )
        sig_h = max(sig_h, 1e-3)
        if self.fuse_altitude:
            return np.diag([sig_h**2, sig_h**2, (self.gamma_z * sig_h) ** 2])
        return np.diag([sig_h**2, sig_h**2])


@dataclass
class UpdateInfo:
    gated: bool
    nees: float
    innovation: np.ndarray
    sigma_h: float


class ErrorStateEKF:
    """Loosely-coupled error-state Kalman filter (see module docstring)."""

    def __init__(self, x0: NominalState, P0: np.ndarray, config: EskfConfig | None = None) -> None:
        self.state = x0.copy()
        self.P = np.asarray(P0, dtype=np.float64).copy()
        if self.P.shape != (15, 15):
            raise ValueError("P0 must be 15x15")
        self.cfg = config or EskfConfig()
        self._g = gravity_enu()

    # -- prediction ---------------------------------------------------------
    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        cfg = self.cfg
        R = quat_to_rot(self.state.quat)
        fb = np.asarray(accel, dtype=np.float64) - self.state.bias_acc
        wb = np.asarray(gyro, dtype=np.float64) - self.state.bias_gyro

        F = np.eye(15)
        F[P_, V_] = np.eye(3) * dt
        F[V_, TH_] = -R @ _skew(fb) * dt
        F[V_, BA_] = -R * dt
        F[TH_, TH_] = np.eye(3) - _skew(wb) * dt
        F[TH_, BG_] = -np.eye(3) * dt

        Q = np.zeros((15, 15))
        Q[V_, V_] = (cfg.accel_noise**2 * dt) * np.eye(3)
        Q[TH_, TH_] = (cfg.gyro_noise**2 * dt) * np.eye(3)
        Q[BA_, BA_] = (cfg.accel_bias_rw**2 * dt) * np.eye(3)
        Q[BG_, BG_] = (cfg.gyro_bias_rw**2 * dt) * np.eye(3)

        self.state = mechanize_step(self.state, gyro, accel, dt, self._g)
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # -- measurement update ----------------------------------------------
    def update(self, z_enu: np.ndarray, confidence: float, spread_m: float) -> UpdateInfo:
        cfg = self.cfg
        m = 3 if cfg.fuse_altitude else 2
        Rk = cfg.r_matrix(confidence, spread_m)
        sigma_h = float(np.sqrt(Rk[0, 0]))

        H = np.zeros((m, 15))
        H[:, 0:m] = np.eye(m)
        h = self.state.pos[:m]
        z = np.asarray(z_enu, dtype=np.float64)[:m]
        r = z - h

        S = H @ self.P @ H.T + Rk
        Sinv = np.linalg.inv(S)
        nees = float(r @ Sinv @ r)
        if nees > chi2_ppf(cfg.gate_prob, m):
            return UpdateInfo(gated=True, nees=nees, innovation=r, sigma_h=sigma_h)

        K = self.P @ H.T @ Sinv
        dx = K @ r

        self.state.pos = self.state.pos + dx[P_]
        self.state.vel = self.state.vel + dx[V_]
        self.state.quat = quat_normalize(quat_mul(self.state.quat, quat_from_rotvec(dx[TH_])))
        self.state.bias_acc = self.state.bias_acc + dx[BA_]
        self.state.bias_gyro = self.state.bias_gyro + dx[BG_]

        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rk @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return UpdateInfo(gated=False, nees=nees, innovation=r, sigma_h=sigma_h)

    # -- convenience ----------------------------------------------------
    def pos_sigma_h(self) -> float:
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def default_p0(
    pos_sigma_m: float = 2.0,
    vert_sigma_m: float = 3.0,
    vel_sigma: float = 0.5,
    yaw_sigma_deg: float = 2.0,
    rp_sigma_deg: float = 5.0,
    ba_sigma: float = 0.1,
    bg_sigma: float = 0.2 * np.pi / 180.0,
) -> np.ndarray:
    """Initial covariance for a known-start launch (design doc §7.4)."""
    d = np.zeros(15)
    d[P_] = [pos_sigma_m**2, pos_sigma_m**2, vert_sigma_m**2]
    d[V_] = vel_sigma**2
    rp = np.radians(rp_sigma_deg) ** 2
    yaw = np.radians(yaw_sigma_deg) ** 2
    d[TH_] = [rp, rp, yaw]
    d[BA_] = ba_sigma**2
    d[BG_] = bg_sigma**2
    return np.diag(d)
