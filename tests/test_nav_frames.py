import unittest

import numpy as np

from avl.nav.frames import (
    LocalFrame,
    euler_from_quat,
    quat_from_euler,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    quat_to_rot,
    rot_to_quat,
    rotvec_from_quat,
    skew,
    slerp,
)


class TestQuaternions(unittest.TestCase):
    def test_skew_is_cross(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a, b = rng.normal(size=3), rng.normal(size=3)
            np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-12)

    def test_euler_roundtrip(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            roll = rng.uniform(-2.9, 2.9)
            pitch = rng.uniform(-1.4, 1.4)  # stay away from gimbal lock
            yaw = rng.uniform(-3.0, 3.0)
            q = quat_from_euler(roll, pitch, yaw)
            r2, p2, y2 = euler_from_quat(q)
            np.testing.assert_allclose([r2, p2, y2], [roll, pitch, yaw], atol=1e-8)

    def test_rot_quat_roundtrip(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            q = quat_normalize(rng.normal(size=4))
            R = quat_to_rot(q)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
            q2 = rot_to_quat(R)
            # q and -q are the same rotation
            self.assertLess(min(np.linalg.norm(q - q2), np.linalg.norm(q + q2)), 1e-8)

    def test_exp_log_roundtrip(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(0.0, 3.0)  # < pi: unique log map
            phi = axis * angle
            q = quat_from_rotvec(phi)
            np.testing.assert_allclose(rotvec_from_quat(q), phi, atol=1e-8)

    def test_rotvec_matches_rotation(self):
        phi = np.array([0.0, 0.0, np.pi / 2])
        R = quat_to_rot(quat_from_rotvec(phi))
        np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-9)

    def test_quat_mul_composes_rotations(self):
        rng = np.random.default_rng(4)
        qa = quat_normalize(rng.normal(size=4))
        qb = quat_normalize(rng.normal(size=4))
        np.testing.assert_allclose(
            quat_to_rot(quat_mul(qa, qb)), quat_to_rot(qa) @ quat_to_rot(qb), atol=1e-10
        )

    def test_slerp_endpoints_and_midpoint(self):
        q0 = quat_from_euler(0, 0, 0.0)
        q1 = quat_from_euler(0, 0, 1.0)
        np.testing.assert_allclose(quat_to_rot(slerp(q0, q1, 0.0)), quat_to_rot(q0), atol=1e-9)
        np.testing.assert_allclose(quat_to_rot(slerp(q0, q1, 1.0)), quat_to_rot(q1), atol=1e-9)
        mid = euler_from_quat(slerp(q0, q1, 0.5))
        self.assertAlmostEqual(mid[2], 0.5, places=6)


class TestLocalFrame(unittest.TestCase):
    def test_geo_enu_roundtrip_region10(self):
        # region 10 mosaic corners (data/UAVVisLoc/satellite_ coordinates_range.csv)
        lf = LocalFrame(lat0=40.355093, lon0=115.776356, alt0=772.0)
        rng = np.random.default_rng(0)
        lat = 40.355093 + rng.uniform(-0.02, 0.0, size=50)
        lon = 115.776356 + rng.uniform(0.0, 0.02, size=50)
        alt = 772.0 + rng.uniform(-50, 50, size=50)
        enu = lf.geo_to_enu(lat, lon, alt)
        lat2, lon2, alt2 = lf.enu_to_geo(enu[:, 0], enu[:, 1], enu[:, 2])
        # round-trip through metres must be sub-millimetre
        back = lf.geo_to_enu(lat2, lon2, alt2)
        np.testing.assert_allclose(back, enu, atol=1e-6)
        np.testing.assert_allclose(alt2, alt, atol=1e-6)

    def test_origin_maps_to_zero(self):
        lf = LocalFrame(40.35, 115.78, 100.0)
        np.testing.assert_allclose(lf.geo_to_enu(40.35, 115.78, 100.0), [0, 0, 0], atol=1e-9)

    def test_east_north_signs(self):
        lf = LocalFrame(40.0, 115.0, 0.0)
        east = lf.geo_to_enu(40.0, 115.01, 0.0)
        north = lf.geo_to_enu(40.01, 115.0, 0.0)
        self.assertGreater(east[0], 100.0)
        self.assertAlmostEqual(east[1], 0.0, places=6)
        self.assertGreater(north[1], 100.0)
        self.assertAlmostEqual(north[0], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
