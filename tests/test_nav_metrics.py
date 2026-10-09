import unittest

import numpy as np

from avl.nav.metrics import along_cross_track, horizontal_error, path_length, trajectory_metrics


class TestMetrics(unittest.TestCase):
    def test_horizontal_error_ignores_altitude(self):
        gt = np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 120.0]])
        est = np.array([[3.0, 4.0, 999.0], [10.0, 0.0, -50.0]])
        np.testing.assert_allclose(horizontal_error(est, gt), [5.0, 0.0])

    def test_cross_track_on_eastward_leg(self):
        t = np.linspace(0, 100, 101)
        gt = np.stack([t, np.zeros_like(t), np.full_like(t, 50.0)], axis=1)
        est = gt.copy()
        est[:, 1] += 3.0  # shifted 3 m north of an eastbound track
        along, cross = along_cross_track(est, gt)
        np.testing.assert_allclose(cross, 3.0, atol=1e-6)
        np.testing.assert_allclose(along, 0.0, atol=1e-6)

    def test_path_length_straight_line(self):
        gt = np.stack([np.linspace(0, 300, 50), np.zeros(50), np.zeros(50)], axis=1)
        self.assertAlmostEqual(path_length(gt), 300.0, places=6)

    def test_trajectory_metrics_shape_and_drift(self):
        t = np.linspace(0, 100, 101)
        gt = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        est = gt.copy()
        est[:, 1] += np.linspace(0, 10, 101)  # error grows to 10 m at the end
        m = trajectory_metrics(est, gt, n_frames=101, n_fix_accepted=90, n_fix_gated=3)
        self.assertAlmostEqual(m["final_error_m"], 10.0, places=6)
        self.assertAlmostEqual(m["drift_rate_pct"], 10.0, places=4)  # 10 m / 100 m
        self.assertAlmostEqual(m["fix_availability"], 90 / 101, places=6)
        self.assertEqual(m["fixes_gated"], 3)
        for key in ("mean", "median", "p95", "min", "max"):
            self.assertIn(key, m["horiz_error_m"])


if __name__ == "__main__":
    unittest.main()
