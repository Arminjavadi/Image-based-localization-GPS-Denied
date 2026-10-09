import unittest

import numpy as np

from avl.nav.frames import GRAVITY, euler_from_quat, quat_from_euler
from avl.nav.ins import NominalState, mechanize_step


LEVEL_SF = np.array([0.0, 0.0, GRAVITY])  # specific force of a stationary, level body


class TestMechanization(unittest.TestCase):
    def test_stationary_stays_put(self):
        st = NominalState(pos=np.array([10.0, -5.0, 100.0]))
        for _ in range(1000):
            st = mechanize_step(st, np.zeros(3), LEVEL_SF, 0.01)
        np.testing.assert_allclose(st.pos, [10.0, -5.0, 100.0], atol=1e-6)
        np.testing.assert_allclose(st.vel, 0.0, atol=1e-6)

    def test_constant_east_acceleration(self):
        st = NominalState()
        dt, n = 0.001, 5000  # 5 s
        acc = LEVEL_SF + np.array([2.0, 0.0, 0.0])
        for _ in range(n):
            st = mechanize_step(st, np.zeros(3), acc, dt)
        t = dt * n
        np.testing.assert_allclose(st.pos, [0.5 * 2.0 * t * t, 0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(st.vel, [2.0 * t, 0.0, 0.0], atol=1e-6)

    def test_constant_yaw_rate(self):
        st = NominalState(quat=quat_from_euler(0, 0, 0.0))
        dt, n = 0.001, 3000  # 3 s
        w = np.array([0.0, 0.0, 0.2])  # rad/s about up
        for _ in range(n):
            st = mechanize_step(st, w, LEVEL_SF, dt)
        _, _, yaw = euler_from_quat(st.quat)
        self.assertAlmostEqual(yaw, 0.2 * dt * n, places=4)
        np.testing.assert_allclose(st.pos, 0.0, atol=1e-4)

    def test_gyro_bias_is_removed(self):
        st = NominalState(bias_gyro=np.array([0.0, 0.0, 0.2]))
        dt, n = 0.001, 3000
        # measured rate equals the bias -> true rate is zero -> no rotation
        for _ in range(n):
            st = mechanize_step(st, np.array([0.0, 0.0, 0.2]), LEVEL_SF, dt)
        _, _, yaw = euler_from_quat(st.quat)
        self.assertAlmostEqual(yaw, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
