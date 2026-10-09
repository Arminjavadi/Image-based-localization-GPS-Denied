import unittest

import numpy as np

from avl.nav.eskf import ErrorStateEKF, EskfConfig, chi2_ppf, default_p0
from avl.nav.filters import Fix, run_filter
from avl.nav.imu_sim import PRESETS, simulate_imu
from avl.nav.ins import NominalState, mechanize
from avl.nav.metrics import path_length, trajectory_metrics
from tests.nav_helpers import analytic_traj, analytic_vel, frame_times, yaw_along_velocity


def _build(duration=120.0, spacing=2.0, seed=11, preset="consumer"):
    tf = frame_times(duration, spacing)
    pos = analytic_traj(tf)
    yaw = yaw_along_velocity(tf)
    imu = simulate_imu(tf, pos, yaw_deg=yaw, rate_hz=100.0, preset=PRESETS[preset], seed=seed)
    v0 = analytic_vel(np.array([0.0]))[0]
    x0 = NominalState(pos=pos[0].copy(), vel=v0.copy(), quat=imu.true_quat[0].copy())
    return tf, pos, imu, x0


class TestChi2Table(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(chi2_ppf(0.997, 2), 11.618, places=3)
        self.assertAlmostEqual(chi2_ppf(0.95, 3), 7.815, places=3)
        with self.assertRaises(ValueError):
            chi2_ppf(0.5, 2)


class TestUpdateGate(unittest.TestCase):
    def test_outlier_is_gated_and_state_unchanged(self):
        x0 = NominalState(pos=np.zeros(3), quat=np.array([1.0, 0, 0, 0]))
        f = ErrorStateEKF(x0, default_p0(), EskfConfig())
        pos_before = f.state.pos.copy()
        info = f.update(np.array([500.0, 500.0]), confidence=0.6, spread_m=40.0)
        self.assertTrue(info.gated)
        np.testing.assert_allclose(f.state.pos, pos_before, atol=1e-12)

    def test_reasonable_fix_pulls_state(self):
        # large initial position uncertainty: one good fix should pull the state
        # most of the way toward it
        x0 = NominalState(pos=np.zeros(3), quat=np.array([1.0, 0, 0, 0]))
        f = ErrorStateEKF(x0, default_p0(pos_sigma_m=60.0), EskfConfig())
        info = f.update(np.array([12.0, -8.0]), confidence=0.9, spread_m=8.0)
        self.assertFalse(info.gated)
        self.assertGreater(f.state.pos[0], 4.0)
        self.assertLess(f.state.pos[1], -2.0)

    def test_covariance_shrinks_on_update(self):
        x0 = NominalState(pos=np.zeros(3), quat=np.array([1.0, 0, 0, 0]))
        f = ErrorStateEKF(x0, default_p0(pos_sigma_m=8.0), EskfConfig())
        p_before = f.P[0, 0]
        f.update(np.array([1.0, 1.0]), confidence=0.9, spread_m=10.0)
        self.assertLess(f.P[0, 0], p_before)


class TestFusionRunLoop(unittest.TestCase):
    def test_fusion_bounds_the_drift(self):
        tf, pos, imu, x0 = _build(duration=120.0, spacing=2.0, seed=11)
        ins = mechanize(imu, x0)
        ins_final = np.linalg.norm(ins.pos[-1] - imu.true_pos[-1])

        rng = np.random.default_rng(99)
        fixes = []
        for k, gi in enumerate(imu.frame_grid_idx):
            if k == 0:
                continue
            noisy = imu.true_pos[gi, :2] + rng.normal(0.0, 8.0, size=2)
            fixes.append(Fix(frame=k, grid_idx=int(gi), pos_enu=noisy,
                             confidence=0.8, spread_m=20.0))

        res = run_filter(imu, fixes, x0, default_p0(), EskfConfig(), "eskf")
        fused = res.at_frames(imu.frame_grid_idx)
        m = trajectory_metrics(
            fused[:, :2], imu.true_pos[imu.frame_grid_idx][:, :2],
            total_path_m=path_length(imu.true_pos[imu.frame_grid_idx]),
            n_frames=len(tf), n_fix_accepted=sum(not g for g in res.fix_gated),
            n_fix_gated=sum(res.fix_gated), nees=res.fix_nees,
        )
        # fused must be far better than unaided INS and inside the 100 m band
        self.assertLess(m["final_error_m"], 0.5 * ins_final)
        self.assertLess(m["horiz_error_m"]["p95"], 25.0)
        self.assertLess(m["horiz_error_m"]["median"], 12.0)

    def test_gate_rejects_injected_outlier_midflight(self):
        tf, pos, imu, x0 = _build(duration=60.0, spacing=2.0, seed=5)
        rng = np.random.default_rng(1)
        fixes = []
        for k, gi in enumerate(imu.frame_grid_idx):
            if k == 0:
                continue
            p = imu.true_pos[gi, :2] + rng.normal(0.0, 6.0, size=2)
            if k == 10:  # one gross outlier
                p = p + np.array([400.0, -300.0])
            fixes.append(Fix(frame=k, grid_idx=int(gi), pos_enu=p,
                             confidence=0.8, spread_m=15.0))
        res = run_filter(imu, fixes, x0, default_p0(), EskfConfig(), "eskf")
        gated_frames = [fr for fr, g in zip(res.fix_frames, res.fix_gated) if g]
        self.assertIn(10, gated_frames)
        fused = res.at_frames(imu.frame_grid_idx)
        # the outlier must not have thrown the solution off
        err = np.linalg.norm(fused[:, :2] - imu.true_pos[imu.frame_grid_idx][:, :2], axis=1)
        self.assertLess(err.max(), 40.0)

    def test_coasting_through_a_dropout(self):
        tf, pos, imu, x0 = _build(duration=150.0, spacing=3.0, seed=8)
        ins = mechanize(imu, x0)
        gt = imu.true_pos[imu.frame_grid_idx]
        ins_err = np.linalg.norm(ins.pos[imu.frame_grid_idx][:, :2] - gt[:, :2], axis=1)

        rng = np.random.default_rng(2)
        fixes = []
        for k, gi in enumerate(imu.frame_grid_idx):
            t_k = tf[k]
            if k == 0 or 75.0 <= t_k <= 105.0:  # 30 s fix dropout
                continue
            p = imu.true_pos[gi, :2] + rng.normal(0.0, 7.0, size=2)
            fixes.append(Fix(frame=k, grid_idx=int(gi), pos_enu=p,
                             confidence=0.8, spread_m=18.0))
        res = run_filter(imu, fixes, x0, default_p0(), EskfConfig(), "eskf")
        fused = res.at_frames(imu.frame_grid_idx)
        in_gap = (tf >= 75.0) & (tf <= 105.0)
        gap_err = np.linalg.norm(fused[in_gap, :2] - gt[in_gap, :2], axis=1)
        # coasting a consumer IMU for 30 s: error stays inside the 100 m band and
        # far below what the same unaided INS accumulates over that span
        self.assertLess(gap_err.max(), 80.0)
        self.assertLess(gap_err.max(), 0.25 * ins_err[in_gap].max())


if __name__ == "__main__":
    unittest.main()


class TestResolveHook(unittest.TestCase):
    def test_resolve_sees_live_prediction_and_can_skip(self):
        tf, pos, imu, x0 = _build(duration=60.0, spacing=2.0, seed=5)
        fixes = [Fix(frame=k, grid_idx=int(gi), pos_enu=np.zeros(2), confidence=0.8, spread_m=20.0)
                 for k, gi in enumerate(imu.frame_grid_idx) if k > 0]
        seen = []

        def resolve(f, filt):
            seen.append(np.linalg.norm(filt.state.pos[:2] - imu.true_pos[f.grid_idx, :2]))
            if f.frame % 2:
                return None  # odd frames: no measurement
            return Fix(frame=f.frame, grid_idx=f.grid_idx,
                       pos_enu=imu.true_pos[f.grid_idx, :2], confidence=0.8, spread_m=20.0)

        res = run_filter(imu, fixes, x0, default_p0(), EskfConfig(), "eskf", resolve=resolve)
        self.assertEqual(len(seen), len(fixes))
        self.assertEqual(res.fix_frames, [f.frame for f in fixes if f.frame % 2 == 0])
        # the prediction handed to resolve is the propagated state, not the placeholder
        self.assertLess(max(seen), 50.0)
