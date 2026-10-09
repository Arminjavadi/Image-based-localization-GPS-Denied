import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from avl.config import AVLConfig
from avl.rerank import (
    RerankConfig,
    Reranker,
    _is_plausible_homography,
    apply_rerank,
    build_reranker,
    load_gray,
)


def _texture(seed: int, size: int = 480) -> np.ndarray:
    """A blurred noise field: plenty of stable corners, no repeated structure."""
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (size, size), dtype=np.uint8), (5, 5), 0)


class RerankFixture(unittest.TestCase):
    """One scene photographed twice, plus unrelated scenes to reject."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)

        scene = _texture(0, 600)
        cls.true_path = root / "true.png"
        Image.fromarray(scene).save(cls.true_path)

        warp = cv2.getRotationMatrix2D((300, 300), 15.0, 1.1)
        query = cv2.warpAffine(scene, warp, (600, 600))
        query = np.clip(query.astype(np.float32) * 1.2 + 10, 0, 255).astype(np.uint8)
        cls.query_path = root / "query.png"
        Image.fromarray(query).save(cls.query_path)

        cls.distractors = []
        for i in range(3):
            path = root / f"other{i}.png"
            Image.fromarray(_texture(100 + i, 600)).save(path)
            cls.distractors.append(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def candidates(self):
        """True match placed last, so only geometry can promote it."""
        return [*self.distractors, self.true_path]

    def descriptor_scores(self):
        """Descriptor ranks the true match worst."""
        return [0.95, 0.92, 0.90, 0.70]


class TestDisabled(RerankFixture):
    def test_disabled_reranker_preserves_order(self):
        reranker = Reranker(RerankConfig(enabled=False))
        outcome = reranker.rerank(
            self.query_path, self.candidates(), self.descriptor_scores()
        )
        self.assertTrue(outcome.skipped)
        self.assertEqual(outcome.order, [0, 1, 2, 3])
        self.assertEqual(outcome.verified, 0)
        # Fusion must still see the descriptor scores, not a row of zeros.
        for stat, score in zip(outcome.stats, self.descriptor_scores()):
            self.assertAlmostEqual(stat.fusion_weight, score)

    def test_apply_rerank_without_reranker_is_a_trim(self):
        ranked = apply_rerank(
            None, self.query_path, [7, 8, 9, 10], self.descriptor_scores(),
            self.candidates(), top_k=2,
        )
        np.testing.assert_array_equal(ranked.indices, [7, 8])
        np.testing.assert_allclose(ranked.weights, [0.95, 0.92])
        self.assertIsNone(ranked.summary)
        self.assertEqual(ranked.per_match, [None, None])

    def test_disabled_via_apply_rerank_matches_descriptor_order(self):
        ranked = apply_rerank(
            Reranker(RerankConfig(enabled=False)), self.query_path,
            [0, 1, 2, 3], self.descriptor_scores(), self.candidates(), top_k=4,
        )
        np.testing.assert_array_equal(ranked.indices, [0, 1, 2, 3])


class TestPromotion(RerankFixture):
    def test_true_match_is_promoted_over_stronger_descriptors(self):
        for backend in ("sift-ransac", "orb-ransac", "akaze-ransac"):
            with self.subTest(backend=backend):
                reranker = Reranker(RerankConfig(enabled=True, backend=backend))
                outcome = reranker.rerank(
                    self.query_path, self.candidates(), self.descriptor_scores()
                )
                self.assertEqual(
                    outcome.order[0], 3, f"{backend} failed to promote the true match"
                )
                self.assertEqual(outcome.verified, 1)
                self.assertTrue(outcome.stats[3].verified)
                self.assertFalse(any(outcome.stats[i].verified for i in range(3)))

    def test_apply_rerank_reports_movement_and_weights(self):
        reranker = Reranker(RerankConfig(enabled=True, backend="sift-ransac"))
        ranked = apply_rerank(
            reranker, self.query_path, [10, 11, 12, 13],
            self.descriptor_scores(), self.candidates(), top_k=2,
        )
        self.assertEqual(int(ranked.indices[0]), 13)
        self.assertEqual(ranked.retrieval_ranks[0], 4)
        self.assertTrue(ranked.per_match[0]["verified"])
        # A rejected candidate must not outweigh the verified winner in fusion,
        # however strong its descriptor score was.
        self.assertGreater(ranked.weights[0], ranked.weights[1])
        self.assertEqual(ranked.weights[1], 0.0)
        self.assertEqual(ranked.summary["backend"], "sift-ransac")
        self.assertIsNone(ranked.summary["error"])

    def test_rejected_candidates_lose_their_fusion_weight(self):
        reranker = Reranker(RerankConfig(enabled=True))
        outcome = reranker.rerank(
            self.query_path, self.candidates(), self.descriptor_scores()
        )
        self.assertGreater(outcome.stats[3].fusion_weight, 0.0)
        for i in range(3):
            self.assertEqual(outcome.stats[i].fusion_weight, 0.0)

    def test_weights_fall_back_to_descriptors_when_nothing_verifies(self):
        reranker = Reranker(RerankConfig(enabled=True))
        outcome = reranker.rerank(self.query_path, self.distractors, [0.9, 0.8, 0.7])
        self.assertEqual(outcome.verified, 0)
        for stat in outcome.stats:
            self.assertAlmostEqual(stat.fusion_weight, stat.descriptor_score)

    def test_blend_zero_keeps_descriptor_order_among_verified(self):
        reranker = Reranker(RerankConfig(enabled=True, blend=0.0))
        outcome = reranker.rerank(
            self.query_path, self.candidates(), self.descriptor_scores()
        )
        # Verification still gates, but scores are untouched by geometry.
        for stat in outcome.stats:
            self.assertAlmostEqual(stat.combined_score, stat.descriptor_score)


class TestAbstains(RerankFixture):
    """The property that matters: the stage helps or abstains, never scrambles."""

    def test_order_is_unchanged_when_nothing_verifies(self):
        reranker = Reranker(RerankConfig(enabled=True, backend="sift-ransac"))
        outcome = reranker.rerank(
            self.query_path, self.distractors, [0.95, 0.92, 0.90]
        )
        self.assertEqual(outcome.verified, 0)
        self.assertEqual(outcome.order, [0, 1, 2])

    def test_unreachable_min_inliers_falls_back_to_descriptor_order(self):
        reranker = Reranker(RerankConfig(enabled=True, min_inliers=10**6))
        outcome = reranker.rerank(
            self.query_path, self.candidates(), self.descriptor_scores()
        )
        self.assertEqual(outcome.verified, 0)
        self.assertEqual(outcome.order, [0, 1, 2, 3])

    def test_missing_optional_backend_degrades_instead_of_raising(self):
        reranker = Reranker(RerankConfig(enabled=True, backend="loftr"))
        try:
            import kornia  # noqa: F401
        except ImportError:
            outcome = reranker.rerank(
                self.query_path, self.candidates(), self.descriptor_scores()
            )
            self.assertTrue(outcome.skipped)
            self.assertIn("kornia", (outcome.error or "").lower())
            self.assertEqual(outcome.order, [0, 1, 2, 3])
            # A missing backend must not silently zero out the fusion weights.
            for stat, score in zip(outcome.stats, self.descriptor_scores()):
                self.assertAlmostEqual(stat.fusion_weight, score)
        else:
            self.skipTest("kornia is installed; the fallback path cannot be exercised")

    def test_unreadable_candidate_does_not_sink_the_query(self):
        missing = Path(self._tmp.name) / "does_not_exist.png"
        reranker = Reranker(RerankConfig(enabled=True))
        outcome = reranker.rerank(
            self.query_path, [missing, self.true_path], [0.9, 0.6]
        )
        self.assertIsNotNone(outcome.stats[0].error)
        self.assertEqual(outcome.order[0], 1)


class TestHomographyGuard(unittest.TestCase):
    def test_identity_is_plausible(self):
        self.assertTrue(_is_plausible_homography(np.eye(3), (480, 640)))

    def test_collapse_to_a_point_is_rejected(self):
        # The exact failure RANSAC produces on random cross-domain matches.
        collapsed = np.array(
            [[0.0, 0.0, 279.4], [0.0, 0.0, 386.1], [0.0, 0.0, 1.0]]
        )
        self.assertFalse(_is_plausible_homography(collapsed, (480, 640)))

    def test_folded_quad_is_rejected(self):
        # A pure mirror stays a convex quad and is a legitimate warp.
        flip = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(_is_plausible_homography(flip, (480, 640)))
        # The horizon line crossing the frame folds it into a bowtie: two corners
        # land behind the camera and the outline self-intersects.
        folded = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0 / 240.0, 1.0]])
        self.assertFalse(_is_plausible_homography(folded, (480, 640)))

    def test_mild_perspective_is_accepted(self):
        keystone = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0004, 1.0]])
        self.assertTrue(_is_plausible_homography(keystone, (480, 640)))

    def test_extreme_zoom_is_rejected(self):
        tiny = np.diag([0.005, 0.005, 1.0])
        huge = np.diag([200.0, 200.0, 1.0])
        self.assertFalse(_is_plausible_homography(tiny, (480, 640)))
        self.assertFalse(_is_plausible_homography(huge, (480, 640)))

    def test_non_finite_is_rejected(self):
        self.assertFalse(_is_plausible_homography(np.full((3, 3), np.nan), (480, 640)))
        self.assertFalse(_is_plausible_homography(None, (480, 640)))


class TestConfig(unittest.TestCase):
    def test_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            RerankConfig(backend="not-a-matcher")
        with self.assertRaises(ValueError):
            AVLConfig(rerank_backend="not-a-matcher")

    def test_rejects_out_of_range_blend(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                RerankConfig(blend=bad)
            with self.assertRaises(ValueError):
                AVLConfig(rerank_blend=bad)

    def test_rejects_empty_candidate_depth(self):
        with self.assertRaises(ValueError):
            RerankConfig(candidates=0)
        with self.assertRaises(ValueError):
            AVLConfig(rerank_candidates=0)

    def test_search_depth_follows_the_toggle(self):
        off = AVLConfig(top_k=5, rerank_candidates=25)
        self.assertEqual(off.rerank_search_k(), 5)
        on = AVLConfig(top_k=5, rerank_candidates=25, rerank_enabled=True)
        self.assertEqual(on.rerank_search_k(), 25)
        # top_k always wins when it is the deeper of the two.
        deep = AVLConfig(top_k=40, rerank_candidates=10, rerank_enabled=True)
        self.assertEqual(deep.rerank_search_k(), 40)

    def test_built_from_avl_config(self):
        config = AVLConfig(
            rerank_enabled=True, rerank_backend="orb-ransac", rerank_blend=0.25
        )
        reranker = build_reranker(config)
        self.assertTrue(reranker.enabled)
        self.assertEqual(reranker.config.backend, "orb-ransac")
        self.assertAlmostEqual(reranker.config.blend, 0.25)

    def test_defaults_leave_the_pipeline_untouched(self):
        self.assertFalse(AVLConfig().rerank_enabled)
        self.assertFalse(build_reranker(AVLConfig()).enabled)


class TestLoadGray(unittest.TestCase):
    def test_downscales_only_when_oversized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.png"
            Image.fromarray(_texture(9, 200)).resize((900, 300)).save(path)
            capped = load_gray(path, 640)
            self.assertEqual(max(capped.shape), 640)
            self.assertEqual(capped.shape, (213, 640))
            untouched = load_gray(path, 4000)
            self.assertEqual(untouched.shape, (300, 900))

    def test_accepts_a_pil_image(self):
        gray = load_gray(Image.fromarray(_texture(11, 300)).convert("RGB"), 128)
        self.assertEqual(gray.shape, (128, 128))
        self.assertEqual(gray.dtype, np.uint8)


class TestFeatureCache(RerankFixture):
    def test_repeated_candidates_reuse_prepared_features(self):
        reranker = Reranker(RerankConfig(enabled=True))
        reranker.rerank(self.query_path, self.candidates(), self.descriptor_scores())
        self.assertEqual(len(reranker._cache), len(self.candidates()))
        first = reranker._cache[str(self.true_path)]
        reranker.rerank(self.query_path, self.candidates(), self.descriptor_scores())
        self.assertIs(reranker._cache[str(self.true_path)], first)

    def test_cache_is_bounded(self):
        reranker = Reranker(RerankConfig(enabled=True, cache_size=2))
        reranker.rerank(self.query_path, self.candidates(), self.descriptor_scores())
        self.assertLessEqual(len(reranker._cache), 2)


if __name__ == "__main__":
    unittest.main()
