import math
import unittest

import numpy as np
from PIL import Image

from avl.retrieval import query_variants, rotate_no_padding


class TestRotateNoPadding(unittest.TestCase):
    def test_no_padding_at_any_angle(self):
        image = Image.new("RGB", (600, 400), (200, 100, 50))
        for angle in (0, 17, 45, 90, 133, -60, 180, 271):
            out = rotate_no_padding(image, angle)
            self.assertEqual(out.size[0], out.size[1])
            self.assertFalse((np.asarray(out).sum(axis=2) == 0).any(), angle)
            t = math.radians(angle)
            expected = 400 / (abs(math.cos(t)) + abs(math.sin(t)))
            self.assertLessEqual(abs(out.size[0] - expected), 4)

    def test_variant_count_and_order(self):
        image = Image.new("RGB", (300, 200))
        variants = query_variants(image, 4, "square", scales=(1.0, 0.5), heading_deg=30.0)
        self.assertEqual(len(variants), 8)
        self.assertGreater(variants[0].size[0], variants[4].size[0])  # scale-major


if __name__ == "__main__":
    unittest.main()
