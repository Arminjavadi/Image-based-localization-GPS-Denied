"""Coordinate frames, gravity, and quaternion algebra for the visual-inertial filter.

Conventions (see docs/VisualInertial_AVL_Design.md §4):

* Navigation frame ``n`` is a local **ENU** tangent plane, origin at a fixed
  reference geodetic point. Axes: x = east, y = north, z = up (metres).
* Body frame ``b`` is x-forward, y-left, z-up.
* Quaternions are scalar-first ``q = [w, x, y, z]``, unit norm, and rotate a
  **body** vector into the **navigation** frame: ``v_n = R{q} @ v_b``.
* Euler angles are the ZYX (yaw-pitch-roll) sequence: yaw about body z, then
  pitch about body y, then roll about body x.

Pure numpy, no other project imports, so ``avl.nav`` stays unit-testable in
isolation and free of torch / Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.80665  # m/s^2, standard gravity


def gravity_enu() -> np.ndarray:
    """Gravity vector in the ENU navigation frame (points down ⇒ -z)."""
    return np.array([0.0, 0.0, -GRAVITY], dtype=np.float64)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix ``[v]_x`` such that ``[v]_x @ w == cross(v, w)``."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# quaternion algebra (scalar-first, body -> nav)
# --------------------------------------------------------------------------- #
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    return q if q[0] >= 0.0 else -q  # canonical: non-negative scalar part


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R{q} with ``v_n = R @ v_b``."""
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def quat_from_rotvec(phi: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector (axis * angle, rad) -> quaternion."""
    phi = np.asarray(phi, dtype=np.float64)
    angle = float(np.linalg.norm(phi))
    if angle < 1e-9:
        q = np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]])
        return quat_normalize(q)
    axis = phi / angle
    s = np.sin(0.5 * angle)
    return np.array([np.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s])


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """Logarithm map: quaternion -> rotation vector (rad)."""
    q = quat_normalize(q)
    w = float(np.clip(q[0], -1.0, 1.0))
    v = q[1:]
    vn = float(np.linalg.norm(v))
    if vn < 1e-9:
        return 2.0 * v  # small-angle
    angle = 2.0 * np.arctan2(vn, w)
    return (angle / vn) * v


def _quat_axis(angle: float, axis: int) -> np.ndarray:
    q = np.zeros(4)
    q[0] = np.cos(0.5 * angle)
    q[1 + axis] = np.sin(0.5 * angle)
    return q


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler angles (rad) -> body->nav quaternion."""
    qz = _quat_axis(yaw, 2)
    qy = _quat_axis(pitch, 1)
    qx = _quat_axis(roll, 0)
    return quat_normalize(quat_mul(quat_mul(qz, qy), qx))


def euler_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    """body->nav quaternion -> (roll, pitch, yaw) in rad, ZYX convention."""
    w, x, y, z = quat_normalize(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation, ``t`` in [0, 1]."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    s0 = np.sin((1.0 - t) * theta0) / np.sin(theta0)
    s1 = np.sin(t * theta0) / np.sin(theta0)
    return quat_normalize(s0 * q0 + s1 * q1)


# --------------------------------------------------------------------------- #
# local ENU tangent plane  <->  geodetic
# --------------------------------------------------------------------------- #
def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """WGS-84 series for metres per degree of latitude / longitude at ``lat_deg``.

    Identical to ``avl.geo._meters_per_degree``; duplicated here to keep
    ``avl.nav`` free of cross-imports.
    """
    lat = np.radians(lat_deg)
    m_lat = 111_132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat)
    m_lon = 111_412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat)
    return float(m_lat), float(m_lon)


@dataclass(frozen=True)
class LocalFrame:
    """Equirectangular ENU tangent plane about a fixed geodetic origin.

    Valid for the < ~10 km spans of a UAV-VisLoc sortie: the linearisation error
    over that range is well under 0.1 m.
    """

    lat0: float
    lon0: float
    alt0: float = 0.0

    @property
    def _mpd(self) -> tuple[float, float]:
        return meters_per_degree(self.lat0)

    def geo_to_enu(self, lat, lon, alt=None) -> np.ndarray:
        m_lat, m_lon = self._mpd
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        east = (lon - self.lon0) * m_lon
        north = (lat - self.lat0) * m_lat
        if alt is None:
            up = np.zeros_like(east)
        else:
            up = np.asarray(alt, dtype=np.float64) - self.alt0
        return np.stack([east, north, up], axis=-1)

    def enu_to_geo(self, east, north, up=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m_lat, m_lon = self._mpd
        east = np.asarray(east, dtype=np.float64)
        north = np.asarray(north, dtype=np.float64)
        lat = self.lat0 + north / m_lat
        lon = self.lon0 + east / m_lon
        alt = self.alt0 + np.asarray(up, dtype=np.float64)
        return lat, lon, alt
