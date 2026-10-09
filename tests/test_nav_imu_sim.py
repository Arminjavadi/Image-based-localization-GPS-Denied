import unittest

import numpy as np

from avl.nav.frames import quat_from_euler
from avl.nav.imu_sim import PRESETS, CubicSpline1D, GradePreset, simulate_imu
from avl.nav.ins import NominalState, mechanize
from tests.nav_helpers import analytic_traj, analytic_vel, frame_times, yaw_along_velocity


class TestCubicSpline(unittest.TestCase):
    def test_interpolates_knots_exactly(self):
        x = np.array([0.0, 1.0, 2.5, 4.0, 7.0])
        y = np.array([0.0, 1.0, -1.0, 2.0, 0.5])
        s = CubicSpline1D(x, y)
        np.testing.assert_allclose(s(x), y, atol=1e-10)

    def test_reproduces_cubic_polynomial(self):
        # a cubic is closely tracked by its natural spline away from the ends
        # (natural BCs force y''=0 at the boundary, so keep the check interior)
        x = np.linspace(0, 10, 41)
        poly = lambda t: 0.3 * t**3 - 1.2 * t**2 + 2.0 * t + 5.0
        s = CubicSpline1D(x, poly(x))
        tt = np.linspace(2.0, 8.0, 61)
        np.testing.assert_allclose(s(tt), poly(tt), atol=2e-2)
        np.testing.assert_allclose(s(tt, 1), 0.9 * tt**2 - 2.4 * tt + 2.0, atol=0.15)
        np.testing.assert_allclose(s(tt, 2), 1.8 * tt - 2.4, atol=1.0)


class TestSimulateImu(unittest.TestCase):
    def test_perfect_imu_reconstructs_trajectory(self):
        tf = frame_times(60.0, 3.0)
        pos = analytic_traj(tf)
        yaw = yaw_along_velocity(tf)
        imu = simulate_imu(tf, pos, yaw_deg=yaw, rate_hz=100.0,
                           preset=GradePreset.perfect(), seed=0)

        v0 = analytic_vel(np.array([0.0]))[0]
        x0 = NominalState(pos=pos[0].copy(), vel=v0.copy(), quat=imu.true_quat[0].copy())
        traj = mechanize(imu, x0)

        err = np.linalg.norm(traj.pos - imu.true_pos, axis=1)
        # noise-free: only integration discretisation error over 60 s
        self.assertLess(err.max(), 1.0)
        self.assertLess(err[-1], 1.0)

    def test_noise_free_gyro_matches_true(self):
        tf = frame_times(30.0, 3.0)
        pos = analytic_traj(tf)
        imu = simulate_imu(tf, pos, yaw_deg=yaw_along_velocity(tf),
                           preset=GradePreset.perfect(), seed=1)
        np.testing.assert_allclose(imu.gyro, imu.true_gyro, atol=1e-12)
        np.testing.assert_allclose(imu.accel, imu.true_accel, atol=1e-12)

    def test_stationary_level_body_reads_gravity(self):
        tf = np.arange(0.0, 10.0 + 1e-9, 2.0)
        pos = np.tile([5.0, 5.0, 100.0], (tf.size, 1))
        imu = simulate_imu(tf, pos, yaw_deg=np.zeros(tf.size),
                           preset=GradePreset.perfect(), seed=0)
        expect = np.tile([0.0, 0.0, 9.80665], (len(imu), 1))
        np.testing.assert_allclose(imu.true_accel, expect, atol=1e-6)
        np.testing.assert_allclose(imu.true_gyro, 0.0, atol=1e-9)

    def test_consumer_imu_drifts(self):
        tf = frame_times(120.0, 3.0)
        pos = analytic_traj(tf)
        imu = simulate_imu(tf, pos, yaw_deg=yaw_along_velocity(tf),
                           preset=PRESETS["consumer"], seed=7)
        v0 = analytic_vel(np.array([0.0]))[0]
        x0 = NominalState(pos=pos[0].copy(), vel=v0.copy(), quat=imu.true_quat[0].copy())
        traj = mechanize(imu, x0)
        final_err = np.linalg.norm(traj.pos[-1] - imu.true_pos[-1])
        # a consumer unit unaided for 2 min should be metres-to-hundreds off,
        # and definitely worse than the perfect-IMU sub-metre bound
        self.assertGreater(final_err, 2.0)

    def test_tactical_drifts_less_than_consumer(self):
        tf = frame_times(120.0, 3.0)
        pos = analytic_traj(tf)
        yaw = yaw_along_velocity(tf)
        v0 = analytic_vel(np.array([0.0]))[0]

        def final_drift(preset):
            imu = simulate_imu(tf, pos, yaw_deg=yaw, preset=preset, seed=3)
            x0 = NominalState(pos=pos[0].copy(), vel=v0.copy(), quat=imu.true_quat[0].copy())
            traj = mechanize(imu, x0)
            return np.linalg.norm(traj.pos[-1] - imu.true_pos[-1])

        self.assertLess(final_drift(PRESETS["tactical"]), final_drift(PRESETS["consumer"]))


if __name__ == "__main__":
    unittest.main()
