import unittest

import cv2
import numpy as np

from avl.nav.sparse_vo import SparseVO, nadir_pixel


def _texture(size=(900, 600), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (size[1] // 6, size[0] // 6), dtype=np.uint8), size,
                     interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (3, 3), 0)


class TestSparseVO(unittest.TestCase):
    def setUp(self):
        self.vo = SparseVO(long_side=0)
        big = _texture((1200, 900))
        self.a = big[100:700, 100:1000]
        # the camera moved 60 px right and 40 px up (image axes) between A and B
        self.b = big[60:660, 160:1060]
        self.fa, self.fb = self.vo.features(self.a), self.vo.features(self.b)
        self.size = (900, 600)

    def test_heading_north_maps_image_axes_to_east_north(self):
        s = self.vo.step(self.fa, self.fb, self.size, self.size, gsd_a_m=0.5, heading_a_deg=0.0)
        self.assertTrue(s.ok, s.reason)
        self.assertAlmostEqual(s.east_m, 30.0, delta=1.0)    # 60 px right * 0.5 m
        self.assertAlmostEqual(s.north_m, 20.0, delta=1.0)   # 40 px up -> forward = north
        self.assertAlmostEqual(s.scale, 1.0, delta=0.01)

    def test_heading_east_rotates_the_step(self):
        s = self.vo.step(self.fa, self.fb, self.size, self.size, gsd_a_m=0.5, heading_a_deg=90.0)
        self.assertAlmostEqual(s.east_m, 20.0, delta=1.0)    # forward now points east
        self.assertAlmostEqual(s.north_m, -30.0, delta=1.0)  # right now points south

    def test_scale_check_rejects_wrong_altitude(self):
        s = self.vo.step(self.fa, self.fb, self.size, self.size, 0.5, 0.0, scale_hint=1.5)
        self.assertFalse(s.ok)

    def test_nadir_offset_moves_the_reference_point(self):
        nad = nadir_pixel(self.size, focal_px=900.0, omega_deg=0.0, kappa_deg=0.0)
        self.assertEqual(nad, (450.0, 300.0))
        # the same tilt in A and B cancels out
        tilted = nadir_pixel(self.size, 900.0, 5.0, -3.0)
        s = self.vo.step(self.fa, self.fb, self.size, self.size, 0.5, 0.0, nadir_a=tilted, nadir_b=tilted)
        self.assertAlmostEqual(s.east_m, 30.0, delta=1.0)
        self.assertAlmostEqual(s.north_m, 20.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
