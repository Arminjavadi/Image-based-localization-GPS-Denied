import unittest

import numpy as np

from avl.geo import _from_local_xy, search_window


class TestSearchWindow(unittest.TestCase):
    def setUp(self):
        lat0, lon0 = 40.35, 115.78
        xs, ys = np.meshgrid(np.arange(-1000, 1001, 100.0), np.arange(-1000, 1001, 100.0))
        pts = [_from_local_xy(x, y, lat0, lon0) for x, y in zip(xs.ravel(), ys.ravel())]
        self.lat = np.array([p[0] for p in pts])
        self.lon = np.array([p[1] for p in pts])
        self.dist = np.hypot(xs.ravel(), ys.ravel())
        self.c = (lat0, lon0)

    def test_radius(self):
        mask = search_window(self.lat, self.lon, *self.c, radius_m=250.0)
        np.testing.assert_array_equal(mask, self.dist <= 250.0 + 1e-6)

    def test_min_keep_takes_nearest(self):
        mask = search_window(self.lat, self.lon, *self.c, radius_m=1.0, min_keep=5)
        self.assertEqual(int(mask.sum()), 5)
        self.assertLessEqual(self.dist[mask].max(), 100.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
