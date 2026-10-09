import unittest

import numpy as np

from avl.nav import vo_fusion as vf


def _grid_world(n_frames: int = 60, stride: float = 100.0):
    """A lawnmower-ish track over a 100 m tile grid, 3 s per frame."""
    gx, gy = np.meshgrid(np.arange(-500, 1501, stride), np.arange(-500, 1001, stride))
    refs = np.c_[gx.ravel(), gy.ravel()].astype(float)
    s = np.linspace(0, 1, n_frames)
    gt = np.c_[1000 * s, 300 * np.sin(2 * np.pi * s)]
    t = 3.0 * np.arange(n_frames)
    return gt, t, refs


def _sims(gt, refs, wrong=(), rng=None):
    """Similarity peaked at the true tile; frames in ``wrong`` peak at a random tile
    more than 800 m away (a different one per frame, so they do not agree with
    each other along the track - a constant wrong tile would, and reanchor would
    follow it)."""
    rng = rng or np.random.default_rng(0)
    d = np.hypot(*(gt[:, None] - refs[None]).transpose(2, 0, 1))
    sims = np.exp(-(d / 120.0) ** 2) + 0.01 * rng.random(d.shape)
    for k in wrong:
        sims[k, rng.choice(np.where(d[k] > 800)[0])] = 2.0
    return sims


class TestVoFusion(unittest.TestCase):
    def setUp(self):
        self.gt, self.t, self.refs = _grid_world()
        self.rng = np.random.default_rng(3)

    def _case(self, level, sims, lost=None):
        vo, sig, _ = vf.simulate_vo(self.gt, self.t, level, self.rng, lost)
        return vf.Case(self.gt, self.refs, sims, vo, sig, 100.0)

    def test_perfect_vo_dead_reckons_exactly(self):
        level = vf.VoLevel(0, 0, 0, 0, 0, max_gap_s=np.inf)
        case = self._case(level, _sims(self.gt, self.refs))
        np.testing.assert_allclose(vf.vo_only(case), self.gt, atol=1e-9)

    def test_lost_vo_coast_error_grows_with_time(self):
        level = vf.VoLevel(0, 0, 0, 0, 0, max_gap_s=np.inf, imu_bias_mps2=0.2)
        lost = np.zeros(len(self.gt) - 1, bool)
        lost[10:30] = True
        _, sig, ok = vf.simulate_vo(self.gt, self.t, level, self.rng, lost)
        self.assertFalse(ok[10:30].any())
        self.assertTrue(np.all(np.diff(sig[10:30]) > 0))

    def test_every_method_beats_avl_with_outliers(self):
        wrong = list(range(15, 25))
        level = vf.VO_LEVELS["VIO ~3%"]
        case = self._case(level, _sims(self.gt, self.refs, wrong))
        avl_err = np.hypot(*(vf.avl_top1(case) - self.gt).T)
        self.assertTrue(np.all(avl_err[wrong] > 700))
        for name in ("kf", "kf_reanchor", "kf_window", "pf", "pgo"):
            est = vf.METHODS[name](case, np.random.default_rng(0))
            err = np.hypot(*(est - self.gt).T)
            self.assertLess(np.percentile(err[1:], 95), 150, name)

    def test_reanchor_escapes_a_wrong_lock(self):
        level = vf.VoLevel(0.0, 0.0, 0.0, 0.0, 1.0, max_gap_s=np.inf)
        case = self._case(level, _sims(self.gt, self.refs))
        case.gt = case.gt.copy()
        case.gt[0] += 600.0          # the filters are told a wrong start
        plain = np.hypot(*(vf.kf(case) - self.gt).T)
        fixed = np.hypot(*(vf.kf_reanchor(case) - self.gt).T)
        self.assertGreater(np.median(plain[-20:]), 400)
        self.assertLess(np.median(fixed[-20:]), 100)

    def test_rigid_fit_recovers_transform(self):
        src = self.rng.normal(0, 300, (20, 2))
        R = vf._rot(0.4)
        dst = src @ R.T + np.array([50.0, -20.0])
        Rm, t = vf._rigid_fit(src, dst)
        np.testing.assert_allclose(Rm, R, atol=1e-9)
        np.testing.assert_allclose(t, [50.0, -20.0], atol=1e-6)

    def test_kidnapped_pf_converges(self):
        level = vf.VO_LEVELS["VIO ~1%"]
        case = self._case(level, _sims(self.gt, self.refs))
        est = vf.pf_kidnapped(case, np.random.default_rng(1))
        err = np.hypot(*(est - self.gt).T)
        self.assertLess(np.median(err[20:]), 80)


if __name__ == "__main__":
    unittest.main()
