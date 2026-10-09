"""Fusing visual odometry (VO / VIO) with absolute visual localization (AVL).

Numpy only. Works in a local 2-D ENU plane (metres): the camera's altitude and
attitude are known on board (baro + AHRS), so the open problem is horizontal
position. Each method takes a :class:`Case` and returns the causal per-frame
position estimate (N, 2), i.e. frame ``k`` only uses data up to ``k``.

Inputs per frame ``k``:

* ``sims[k]``  - AVL similarity of the frame to every map tile (``refs``);
* ``vo[k-1]``  - measured displacement from frame ``k-1`` to ``k`` with its
  1-sigma ``vo_sigma[k-1]`` (large where VO lost track).

Methods (see :data:`METHODS`):

``avl_top1`` / ``avl_cluster``  AVL alone (best tile; top-5 cluster fusion).
``vo_only``                     dead reckoning from the known start.
``kf``                          loosely coupled Kalman filter, chi-square gated
                                (the 2-D position core of :mod:`avl.nav.eskf`).
``kf_reanchor``                 ``kf`` + reset onto mutually consistent rejected fixes.
``kf_window``                   ``kf_reanchor`` + search-window prior: AVL only
                                searches tiles within 3 sigma of the prediction.
``pf``                          particle filter weighted by the whole similarity
                                map (keeps several hypotheses alive).
``pgo``                         sliding-window pose graph, Cauchy-robust AVL
                                factors, solved by IRLS (fixed-lag smoother).
``align``                       FoundLoc-style: RANSAC-fit the recent VO track to
                                the AVL fixes (rigid 2-D), no known start.
``pf_kidnapped``                ``pf`` started uniformly over the map.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

CHI2_2D_99 = 9.21


# ── simulated VO ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VoLevel:
    """Frame-to-frame VO error model. Biases are constant over a flight.

    Where vision is lost the step comes from an IMU coast instead: a constant
    unknown acceleration bias (1-sigma ``imu_bias_mps2`` per axis) per lost
    episode, so the coast error grows with the square of the time without vision.
    """

    scale_bias: float       # 1-sigma relative scale bias (altitude / baro error)
    scale_noise: float      # per-step relative scale noise
    yaw_bias_deg: float     # 1-sigma heading bias (compass / AHRS)
    yaw_noise_deg: float    # per-step heading noise
    abs_noise_m: float      # per-step additive noise
    max_gap_s: float = 15.0  # saved frames further apart: sparse VO has no overlap
    imu_bias_mps2: float = 0.2


VO_LEVELS = {
    # continuous VIO on board: tracks straight through the dataset's frame gaps
    "VIO ~1%": VoLevel(0.01, 0.01, 0.5, 0.5, 2.0, max_gap_s=np.inf),
    "VIO ~3%": VoLevel(0.03, 0.03, 2.0, 2.0, 5.0, max_gap_s=np.inf),
    # frame-to-frame VO on the saved frames only, IMU coast across gaps
    "sparse VO ~3%": VoLevel(0.03, 0.03, 2.0, 2.0, 5.0),
    "poor VO ~8%": VoLevel(0.08, 0.08, 5.0, 5.0, 15.0),
}

# nominal: data as recorded.
# avl_outage: AVL "featureless" for 2 x 20 frames - its output is the frame's own
#   scores shuffled over the map (confident-looking garbage); VO still works.
# both_outage: the same blocks with vision lost for VO too (water, uniform sand):
#   only the IMU coast is left there.
SCENARIOS = ("nominal", "avl_outage", "both_outage")
OUTAGE_BLOCKS = ((40, 60), (100, 120))


def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _coast(d_true: np.ndarray, dt: float, tau: float, acc: np.ndarray, bias: float,
           rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """IMU coast over one step that starts ``tau`` s after vision was lost."""
    grow = 0.5 * ((tau + dt) ** 2 - tau**2)
    return d_true + acc * grow + rng.normal(0, 5.0, 2), 1.5 * bias * grow + 10.0


def _clean_dt(gt: np.ndarray, t: np.ndarray, coast_speed_mps: float = 20.0) -> np.ndarray:
    """Time steps; a non-positive one (a log glitch) becomes the time needed to fly
    the step at ``coast_speed_mps``."""
    dt = np.diff(t)
    return np.where(dt > 0, dt, np.linalg.norm(np.diff(gt, axis=0), axis=1) / coast_speed_mps)


def simulate_vo(
    gt: np.ndarray, t: np.ndarray, level: VoLevel, rng: np.random.Generator,
    lost: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measured steps (N-1, 2), the 1-sigma the filters assume (N-1,), VO-ok mask."""
    d_true = np.diff(gt, axis=0)
    dt = _clean_dt(gt, t)
    ok = dt <= level.max_gap_s
    if lost is not None:
        ok &= ~lost
    sb = rng.normal(0, level.scale_bias)
    yb = np.radians(rng.normal(0, level.yaw_bias_deg))
    rel = np.sqrt(level.scale_bias**2 + level.scale_noise**2
                  + np.radians(level.yaw_bias_deg)**2 + np.radians(level.yaw_noise_deg)**2)
    vo = np.zeros_like(d_true)
    sig = np.zeros(len(d_true))
    tau, acc = 0.0, np.zeros(2)            # time without vision, episode bias
    for k, d in enumerate(d_true):
        if ok[k]:
            tau = 0.0
            s = 1 + sb + rng.normal(0, level.scale_noise)
            yaw = yb + np.radians(rng.normal(0, level.yaw_noise_deg))
            vo[k] = s * (_rot(yaw) @ d) + rng.normal(0, level.abs_noise_m, 2)
            sig[k] = np.hypot(rel * np.linalg.norm(vo[k]), level.abs_noise_m)
        else:
            if tau == 0.0:
                acc = rng.normal(0, level.imu_bias_mps2, 2)
            vo[k], sig[k] = _coast(d, dt[k], tau, acc, level.imu_bias_mps2, rng)
            tau += dt[k]
    return vo, sig, ok


@dataclass(frozen=True)
class RealVo:
    """Measured VO steps (``scripts/visloc_vo.py``). Failed steps coast on the IMU.

    The filters assume ``rel_sigma * |step| + floor_m``: one constant for every
    region, not fitted per region.
    """

    steps: np.ndarray        # (N-1, 2) east/north metres, NaN where not ok
    ok: np.ndarray           # (N-1,)
    rel_sigma: float = 0.2
    floor_m: float = 10.0
    imu_bias_mps2: float = 0.2


def real_vo(gt: np.ndarray, t: np.ndarray, level: RealVo, rng: np.random.Generator,
            lost: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_true = np.diff(gt, axis=0)
    dt = _clean_dt(gt, t)
    ok = level.ok.copy()
    if lost is not None:
        ok &= ~lost
    vo = np.zeros_like(d_true)
    sig = np.zeros(len(d_true))
    tau, acc = 0.0, np.zeros(2)
    for k, d in enumerate(d_true):
        if ok[k]:
            tau = 0.0
            vo[k] = level.steps[k]
            sig[k] = level.rel_sigma * np.linalg.norm(vo[k]) + level.floor_m
        else:
            if tau == 0.0:
                acc = rng.normal(0, level.imu_bias_mps2, 2)
            vo[k], sig[k] = _coast(d, dt[k], tau, acc, level.imu_bias_mps2, rng)
            tau += dt[k]
    return vo, sig, ok


@dataclass
class Case:
    gt: np.ndarray        # (N, 2) - only gt[0] (the start) may be used by methods
    refs: np.ndarray      # (M, 2)
    sims: np.ndarray      # (N, M)
    vo: np.ndarray        # (N-1, 2)
    vo_sigma: np.ndarray  # (N-1,)
    tile_stride_m: float

    @property
    def start(self) -> np.ndarray:
        return self.gt[0]


def make_case(gt, t, refs, sims, level: VoLevel | RealVo, scenario: str, rng: np.random.Generator) -> Case:
    sims = sims.copy()
    lost = np.zeros(len(gt) - 1, bool)
    if scenario in ("avl_outage", "both_outage"):
        for a, b in OUTAGE_BLOCKS:
            for k in range(a, min(b, len(gt))):
                sims[k] = rng.permutation(sims[k])
            if scenario == "both_outage":
                lost[a - 1: min(b, len(gt)) - 1] = True
    make = real_vo if isinstance(level, RealVo) else simulate_vo
    vo, sig, _ = make(gt, t, level, rng, lost)
    d = np.hypot(*(refs[:, None] - refs[None]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    return Case(gt, refs, sims, vo, sig, float(np.median(d.min(1))))


# ── AVL alone ──────────────────────────────────────────────────────────────────
def avl_top1(case: Case, rng=None) -> np.ndarray:
    return case.refs[case.sims.argmax(1)]


def _cluster_fix(xy: np.ndarray, scores: np.ndarray, radius: float = 150.0, temp: float = 0.02) -> np.ndarray:
    """avl.geo 'cluster' fusion in the plane: softmax weights, greedy single-linkage."""
    w = np.exp((scores - scores.max()) / temp)
    labels = -np.ones(len(xy), int)
    for i in range(len(xy)):
        if labels[i] < 0:
            labels[i] = i
            stack = [i]
            while stack:
                j = stack.pop()
                near = np.where((labels < 0) & (np.hypot(*(xy - xy[j]).T) <= radius))[0]
                labels[near] = i
                stack.extend(near.tolist())
    best = max(set(labels), key=lambda c: w[labels == c].sum())
    m = labels == best
    return (w[m, None] * xy[m]).sum(0) / w[m].sum()


def avl_cluster(case: Case, rng=None, k: int = 5) -> np.ndarray:
    out = np.zeros((len(case.sims), 2))
    for i, s in enumerate(case.sims):
        top = np.argsort(-s)[:k]
        out[i] = _cluster_fix(case.refs[top], s[top])
    return out


def vo_only(case: Case, rng=None) -> np.ndarray:
    return case.start + np.vstack([np.zeros(2), np.cumsum(case.vo, axis=0)])


# ── Kalman filter (± search window) ────────────────────────────────────────────
def _kf(case: Case, window: bool, reanchor: bool, sigma_avl: float = 50.0, k_sigma: float = 3.0,
        r_min: float = 150.0, n_anchor: int = 4, anchor_m: float = 100.0) -> np.ndarray:
    """Position KF driven by VO.

    ``reanchor``: when the last ``n_anchor`` *global* AVL fixes agree with each
    other once carried along the VO track to the current frame (all within
    ``anchor_m`` of their mean) but that mean is outside the gate around the
    state, the filter is reset onto them. This is the way out of a lock on a
    wrong position. The chi-square gate alone can never leave one, and with a
    search window the windowed fixes always pass the gate, so the test has to
    use the unwindowed fixes.
    """
    x = case.start.astype(float).copy()
    P = np.eye(2) * 10.0**2
    R = np.eye(2) * sigma_avl**2
    out = [x.copy()]
    track = np.vstack([np.zeros(2), np.cumsum(case.vo, axis=0)])
    top1 = case.refs[case.sims.argmax(1)]
    for k in range(1, len(case.sims)):
        x = x + case.vo[k - 1]
        P = P + np.eye(2) * case.vo_sigma[k - 1] ** 2
        s = case.sims[k]
        S = P + R
        if window:
            radius = max(r_min, k_sigma * np.sqrt(np.trace(S) / 2))
            dist = np.hypot(*(case.refs - x).T)
            inside = dist <= radius
            if not inside.any():
                inside[np.argmin(dist)] = True
            z = case.refs[np.where(inside)[0][np.argmax(s[inside])]]
        else:
            z = top1[k]
        r = z - x
        Si = np.linalg.inv(S)
        if r @ Si @ r <= CHI2_2D_99:
            K = P @ Si
            x = x + K @ r
            P = (np.eye(2) - K) @ P
        if reanchor and k >= n_anchor:
            last = np.arange(k - n_anchor + 1, k + 1)
            moved = top1[last] + track[k] - track[last]
            centre = moved.mean(0)
            if np.hypot(*(moved - centre).T).max() <= anchor_m:
                drift = case.vo_sigma[last[0] - 1:k].sum()
                anchor_cov = R / n_anchor + np.eye(2) * drift**2 / n_anchor
                d = centre - x
                if d @ np.linalg.inv(P + anchor_cov) @ d > CHI2_2D_99:
                    x, P = centre, anchor_cov
        out.append(x.copy())
    return np.array(out)


def kf(case: Case, rng=None) -> np.ndarray:
    return _kf(case, window=False, reanchor=False)


def kf_reanchor(case: Case, rng=None) -> np.ndarray:
    return _kf(case, window=False, reanchor=True)


def kf_window(case: Case, rng=None) -> np.ndarray:
    return _kf(case, window=True, reanchor=True)


# ── particle filter over the whole similarity map ──────────────────────────────
def _zscores(s: np.ndarray) -> np.ndarray:
    med = np.median(s)
    mad = np.median(np.abs(s - med)) * 1.4826 + 1e-9
    return (s - med) / mad


def _pf(case: Case, rng: np.random.Generator, kidnapped: bool, n: int = 1000,
        beta: float = 0.5, recover: float = 0.01) -> np.ndarray:
    lo, hi = case.refs.min(0), case.refs.max(0)
    if kidnapped:
        p = rng.uniform(lo, hi, (n, 2))
    else:
        p = case.start + rng.normal(0, 10.0, (n, 2))
    w = np.full(n, 1.0 / n)
    cover = 0.75 * case.tile_stride_m + 25.0      # beyond this a particle is off-map
    out = []
    for k in range(len(case.sims)):
        if k > 0:
            p = p + case.vo[k - 1] + rng.normal(0, max(case.vo_sigma[k - 1], 5.0), (n, 2))
            n_rec = int(recover * n)
            if n_rec:
                idx = rng.choice(n, n_rec, replace=False)
                p[idx] = rng.uniform(lo, hi, (n_rec, 2))
        z = _zscores(case.sims[k])
        d2 = ((p[:, None, :] - case.refs[None]) ** 2).sum(-1)
        nearest = d2.argmin(1)
        zp = np.where(d2[np.arange(n), nearest] <= cover**2, z[nearest], z.min())
        logw = np.log(w + 1e-300) + beta * zp
        w = np.exp(logw - logw.max())
        w /= w.sum()
        out.append(_pf_estimate(p, w, nearest, case.refs))
        if 1.0 / np.sum(w**2) < n / 2:              # systematic resampling
            c = np.cumsum(w)
            u = (rng.random() + np.arange(n)) / n
            p = p[np.minimum(np.searchsorted(c, u), n - 1)]
            w = np.full(n, 1.0 / n)
    return np.array(out)


def _pf_estimate(p: np.ndarray, w: np.ndarray, nearest: np.ndarray, refs: np.ndarray,
                 radius: float = 200.0) -> np.ndarray:
    """Mean of the heaviest mode (not the global mean, which can sit between modes)."""
    tile_w = np.bincount(nearest, weights=w, minlength=len(refs))
    centre = refs[np.argmax(tile_w)]
    m = np.hypot(*(p - centre).T) <= radius
    if w[m].sum() <= 0:                 # the tile's particles all lie beyond radius
        m = nearest == np.argmax(tile_w)
    return (w[m, None] * p[m]).sum(0) / w[m].sum()


def pf(case: Case, rng: np.random.Generator) -> np.ndarray:
    return _pf(case, rng, kidnapped=False)


def pf_kidnapped(case: Case, rng: np.random.Generator) -> np.ndarray:
    return _pf(case, rng, kidnapped=True)


# ── sliding-window robust pose graph ───────────────────────────────────────────
def pgo(case: Case, rng=None, window: int = 15, c_avl: float = 50.0, sigma_avl: float = 50.0,
        sigma_anchor: float = 30.0, iters: int = 10) -> np.ndarray:
    """Fixed-lag smoother: last ``window`` positions, VO between-factors, Cauchy AVL
    factors, and the oldest pose anchored to its previous estimate. IRLS."""
    top1 = case.refs[case.sims.argmax(1)]
    est = [case.start.astype(float).copy()]
    for k in range(1, len(case.sims)):
        a = max(0, k - window + 1)
        idx = np.arange(a, k + 1)
        W = len(idx)
        x = np.vstack([np.asarray(est[a:k]).reshape(-1, 2), est[k - 1] + case.vo[k - 1]])
        anchor_sigma = 5.0 if a == 0 else sigma_anchor
        for _ in range(iters):
            rows, rhs, wts = [], [], []
            e = np.zeros(W); e[0] = 1.0
            rows.append(e); rhs.append(est[a] if a < k else x[0]); wts.append(1 / anchor_sigma**2)
            for j in range(1, W):
                e = np.zeros(W); e[j] = 1.0; e[j - 1] = -1.0
                rows.append(e); rhs.append(case.vo[idx[j] - 1]); wts.append(1 / case.vo_sigma[idx[j] - 1] ** 2)
            for j in range(W):
                if idx[j] == 0:
                    continue
                r = np.linalg.norm(top1[idx[j]] - x[j])
                e = np.zeros(W); e[j] = 1.0
                rows.append(e); rhs.append(top1[idx[j]])
                wts.append(1 / (1 + (r / c_avl) ** 2) / sigma_avl**2)
            A = np.array(rows); b = np.array(rhs); wv = np.array(wts)
            AtW = A.T * wv
            x = np.linalg.solve(AtW @ A + 1e-9 * np.eye(W), AtW @ b)
        est.append(x[-1].copy())
    return np.array(est)


# ── FoundLoc-style trajectory alignment ────────────────────────────────────────
def _rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ms, md = src.mean(0), dst.mean(0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    return Rm, md - Rm @ ms


def align(case: Case, rng: np.random.Generator, window: int = 25, inlier_m: float = 100.0,
          trials: int = 60) -> np.ndarray:
    """Unknown start: rigid-align the last ``window`` VO poses to the AVL fixes with
    RANSAC; output the aligned current pose (AVL top-1 until there is support)."""
    top1 = case.refs[case.sims.argmax(1)]
    track = np.vstack([np.zeros(2), np.cumsum(case.vo, axis=0)])
    out = []
    for k in range(len(case.sims)):
        a = max(0, k - window + 1)
        src, dst = track[a:k + 1], top1[a:k + 1]
        best, best_n = None, 2
        if len(src) >= 3:
            for _ in range(trials):
                i, j = rng.choice(len(src), 2, replace=False)
                if np.linalg.norm(src[i] - src[j]) < 50:
                    continue
                Rm, tv = _rigid_fit(src[[i, j]], dst[[i, j]])
                inl = np.hypot(*((src @ Rm.T + tv) - dst).T) <= inlier_m
                if inl.sum() > best_n:
                    best, best_n = inl, int(inl.sum())
        if best is None:
            out.append(top1[k])
            continue
        Rm, tv = _rigid_fit(src[best], dst[best])
        out.append(Rm @ track[k] + tv)
    return np.array(out)


METHODS: dict[str, Callable[[Case, np.random.Generator], np.ndarray]] = {
    "avl_top1": avl_top1,
    "avl_cluster": avl_cluster,
    "vo_only": vo_only,
    "kf": kf,
    "kf_reanchor": kf_reanchor,
    "kf_window": kf_window,
    "pf": pf,
    "pgo": pgo,
    "align": align,
    "pf_kidnapped": pf_kidnapped,
}
