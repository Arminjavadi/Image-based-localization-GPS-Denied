import unittest

import numpy as np

from avl.nav.raster import RegionRaster, contiguous_run_along_path, snap_path_to_frames


def _raster() -> RegionRaster:
    # region 05 corners from data/UAVVisLoc/satellite_ coordinates_range.csv
    return RegionRaster(
        region="05", tif_path=None,  # type: ignore[arg-type]
        lt_lat=24.666899, lt_lon=102.340055,
        rb_lat=24.650422, rb_lon=102.365252,
        width=9394, height=6144,
    )


class TestGeoPixel(unittest.TestCase):
    def test_corners_map_to_pixel_bounds(self):
        r = _raster()
        x0, y0 = r.geo_to_px(r.lt_lat, r.lt_lon)
        x1, y1 = r.geo_to_px(r.rb_lat, r.rb_lon)
        np.testing.assert_allclose([float(x0), float(y0)], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose([float(x1), float(y1)], [r.width, r.height], atol=1e-6)

    def test_roundtrip(self):
        r = _raster()
        rng = np.random.default_rng(0)
        x = rng.uniform(0, r.width, 40)
        y = rng.uniform(0, r.height, 40)
        lat, lon = r.px_to_geo(x, y)
        x2, y2 = r.geo_to_px(lat, lon)
        np.testing.assert_allclose(x2, x, atol=1e-6)
        np.testing.assert_allclose(y2, y, atol=1e-6)

    def test_axis_orientation(self):
        r = _raster()
        # moving east (higher lon) increases x; moving south (lower lat) increases y
        x_w, _ = r.geo_to_px(r.lt_lat, r.lt_lon)
        x_e, _ = r.geo_to_px(r.lt_lat, r.lt_lon + 0.01)
        _, y_n = r.geo_to_px(r.lt_lat, r.lt_lon)
        _, y_s = r.geo_to_px(r.lt_lat - 0.01, r.lt_lon)
        self.assertGreater(float(x_e), float(x_w))
        self.assertGreater(float(y_s), float(y_n))


class TestSnap(unittest.TestCase):
    def test_orders_by_arclength_and_respects_corridor(self):
        # a straight east-bound path; frames strung along it, two far off to the side
        path = np.array([[0.0, 0.0], [1000.0, 0.0]])
        frame_xy = np.array([
            [100.0, 5.0],     # on path
            [700.0, -8.0],    # on path
            [400.0, 3.0],     # on path
            [500.0, 400.0],   # far -> excluded
            [900.0, 2.0],     # on path
        ])
        ids = ["a", "b", "c", "d", "e"]
        got = snap_path_to_frames(path, frame_xy, ids, corridor_m=50.0)
        self.assertEqual(got, ["a", "c", "b", "e"])

    def test_needs_two_vertices(self):
        self.assertEqual(
            snap_path_to_frames(np.array([[0.0, 0.0]]), np.zeros((3, 2)), ["a", "b", "c"], 10.0),
            [],
        )


class TestContiguousRun(unittest.TestCase):
    def test_longest_run_within_corridor(self):
        path = np.array([[0.0, 0.0], [1000.0, 0.0]])
        # frames 0-1 off, 2-8 on the line, 9 off, 10-11 on
        frame_xy = np.array([[0, 500.0], [0, 400.0]]
                            + [[i * 120.0, 4.0] for i in range(7)]
                            + [[900.0, 600.0], [950.0, 3.0], [980.0, 2.0]])
        run = contiguous_run_along_path(path, frame_xy, corridor_m=40.0, max_gap=0)
        self.assertIsNotNone(run)
        s, e, cov = run
        self.assertEqual((s, e), (2, 8))
        self.assertEqual(cov, 1.0)

    def test_bridges_small_gaps(self):
        path = np.array([[0.0, 0.0], [1000.0, 0.0]])
        frame_xy = np.array([[100.0, 3.0], [200.0, 500.0], [300.0, 3.0], [400.0, 3.0]])
        run = contiguous_run_along_path(path, frame_xy, corridor_m=30.0, max_gap=2)
        self.assertEqual(run[:2], (0, 3))
        self.assertAlmostEqual(run[2], 0.75)  # one of four frames was outside

    def test_none_when_nothing_close(self):
        path = np.array([[0.0, 0.0], [10.0, 0.0]])
        self.assertIsNone(
            contiguous_run_along_path(path, np.array([[0.0, 900.0], [5.0, 900.0]]), 50.0)
        )


if __name__ == "__main__":
    unittest.main()
