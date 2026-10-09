import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from avl.retrieval import centre_crop, rotate_no_padding
from avl.scale import CameraFootprint, agl_crop_fraction, crop_fraction, query_scale, query_square_side_px
from avl.terrain import TerrainModel, cells_for_bounds, tile_name


class TestCameraFootprint(unittest.TestCase):
    def test_hfov_round_trip(self):
        cam = CameraFootprint.from_hfov(60.0, 4000)
        self.assertAlmostEqual(cam.k, 2 * math.tan(math.radians(30)), places=12)
        self.assertAlmostEqual(cam.hfov_deg, 60.0, places=9)

    def test_gsd_scales_with_agl_and_width(self):
        cam = CameraFootprint(1.0, 3000)
        self.assertAlmostEqual(cam.gsd_m(300.0), 0.1)
        self.assertAlmostEqual(cam.gsd_m(600.0), 0.2)
        self.assertAlmostEqual(cam.gsd_m(300.0, width_px=1500), 0.2)


class TestQuerySquare(unittest.TestCase):
    def test_side_matches_rotate_no_padding(self):
        image = Image.new("RGB", (900, 600), (90, 120, 30))
        for heading in (None, 0.0, 17.0, 45.0, 90.0, -133.0):
            expected = query_square_side_px(900, 600, heading)
            if heading is None:
                self.assertEqual(expected, 600)
                continue
            actual = rotate_no_padding(image, heading).size[0]
            self.assertLessEqual(abs(actual - expected), 4, heading)

    def test_45_degrees_loses_a_third_of_the_ground(self):
        self.assertAlmostEqual(query_square_side_px(3000, 2000, 45.0), 2000 / math.sqrt(2), places=6)


class TestCropFraction(unittest.TestCase):
    def test_exact_and_clipped(self):
        # 2000 px at 0.125 m/px = 250 m of ground; a 200 m tile needs 80 % of it
        self.assertAlmostEqual(crop_fraction(2000, 0.125, 200.0), 0.8)
        # the frame covers less than the tile: cropping cannot help
        self.assertEqual(crop_fraction(2000, 0.05, 200.0), 1.0)

    def test_degenerate_inputs_leave_frame_alone(self):
        for gsd in (0.0, -1.0, float("nan"), float("inf")):
            self.assertEqual(crop_fraction(2000, gsd, 200.0), 1.0)

    def test_cropped_square_covers_the_tile(self):
        """End to end: rotate, crop by the fraction, and the ground side equals the tile."""
        cam = CameraFootprint(1.0, 3000)
        image = Image.new("RGB", (3000, 2000))
        for heading in (0.0, 30.0, 45.0, 100.0):
            for agl in (250.0, 400.0, 700.0):
                f, side_m = agl_crop_fraction(cam, agl, 3000, 2000, heading, 150.0)
                view = centre_crop(rotate_no_padding(image, heading), f)
                ground = view.size[0] * cam.gsd_m(agl)
                self.assertLess(abs(ground - min(150.0, side_m)), 2 * cam.gsd_m(agl) + 0.5,
                                (heading, agl, f))


class TestQueryScale(unittest.TestCase):
    def test_falls_back_to_the_full_frame(self):
        cam = CameraFootprint(1.0, 3000)
        self.assertEqual(query_scale(None, 400.0, 3000, 2000, 0.0, 200.0), (1.0,))
        self.assertEqual(query_scale(cam, None, 3000, 2000, 0.0, 200.0), (1.0,))
        self.assertEqual(query_scale(cam, float("nan"), 3000, 2000, 0.0, 200.0), (1.0,))
        self.assertEqual(query_scale(cam, 400.0, 3000, 2000, 0.0, None), (1.0,))
        self.assertEqual(query_scale(cam, 40.0, 3000, 2000, 0.0, 200.0), (1.0,))  # on the ground

    def test_gate_ignores_the_rotation_crop(self):
        cam = CameraFootprint(1.0, 3000)
        # 266.7 m footprint vs 250 m tile = 1.07x: below a 1.25 gate at any heading, even
        # though the 45-degree north-aligned square alone covers only 189 m
        self.assertEqual(query_scale(cam, 400.0, 3000, 2000, 45.0, 250.0, gate=1.25), (1.0,))

    def test_gate_keeps_small_mismatches_uncropped(self):
        cam = CameraFootprint(1.0, 3000)
        # 266.7 m of ground vs a 200 m tile = 1.33x: cropped at gate 1.25, kept at gate 1.5
        self.assertAlmostEqual(query_scale(cam, 400.0, 3000, 2000, None, 200.0, gate=1.25)[0], 0.75)
        self.assertEqual(query_scale(cam, 400.0, 3000, 2000, None, 200.0, gate=1.5), (1.0,))

    def test_informed_crop(self):
        cam = CameraFootprint(1.0, 3000)
        # 400 m AGL -> 0.1333 m/px; 2000 px square = 266.7 m; a 200 m tile needs 75 %
        (f,) = query_scale(cam, 400.0, 3000, 2000, None, 200.0)
        self.assertAlmostEqual(f, 0.75, places=6)


def _write_dem(path: Path, z: np.ndarray, lon0: float, lat0: float, step: float, pixel_is_point: bool):
    import tifffile

    raster_type = 2 if pixel_is_point else 1
    geokeys = [1, 1, 0, 2, 1024, 0, 1, 2, 1025, 0, 1, raster_type]
    tifffile.imwrite(
        path, z.astype(np.float32),
        extratags=[
            (33550, "d", 3, (step, step, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, lon0, lat0, 0.0), False),
            (34735, "H", len(geokeys), geokeys, False),
        ],
    )


class TestTerrain(unittest.TestCase):
    def test_tile_names(self):
        self.assertEqual(tile_name(24, 102), "Copernicus_DSM_COG_10_N24_00_E102_00_DEM")
        self.assertEqual(tile_name(-1, -71), "Copernicus_DSM_COG_10_S01_00_W071_00_DEM")
        self.assertEqual(cells_for_bounds(24.2, 25.1, 101.9, 102.1),
                         [(24, 101), (24, 102), (25, 101), (25, 102)])

    def _check(self, pixel_is_point: bool):
        try:
            import tifffile  # noqa: F401
        except ImportError:
            self.skipTest("tifffile not installed")
        step = 1.0 / 100
        n = 101
        # a tilted plane, which bilinear interpolation reproduces exactly
        if pixel_is_point:
            lat_c = 25.0 - step * np.arange(n)          # tiepoint = centre of pixel (0, 0)
            lon_c = 102.0 + step * np.arange(n)
            tie_lon, tie_lat = 102.0, 25.0
        else:
            lat_c = 25.0 - step * (np.arange(n) + 0.5)  # tiepoint = corner of pixel (0, 0)
            lon_c = 102.0 + step * (np.arange(n) + 0.5)
            tie_lon, tie_lat = 102.0, 25.0
        z = 1000.0 + 300.0 * (lat_c[:, None] - 24.5) + 200.0 * (lon_c[None, :] - 102.5)
        with tempfile.TemporaryDirectory() as tmp:
            _write_dem(Path(tmp) / f"{tile_name(24, 102)}.tif", z, tie_lon, tie_lat, step, pixel_is_point)
            model = TerrainModel(Path(tmp))
            lat = np.array([24.5, 24.137, 24.81])
            lon = np.array([102.5, 102.333, 102.07])
            expected = 1000.0 + 300.0 * (lat - 24.5) + 200.0 * (lon - 102.5)
            np.testing.assert_allclose(model.elevation(lat, lon), expected, atol=1e-3)
            np.testing.assert_allclose(model.agl(expected + 400.0, lat, lon), 400.0, atol=1e-3)
            # a cell that is not on disk gives NaN, not a wrong number
            self.assertTrue(np.isnan(model.elevation([30.5], [102.5])[0]))

    def test_pixel_is_point(self):
        self._check(pixel_is_point=True)

    def test_pixel_is_area(self):
        self._check(pixel_is_point=False)


if __name__ == "__main__":
    unittest.main()
