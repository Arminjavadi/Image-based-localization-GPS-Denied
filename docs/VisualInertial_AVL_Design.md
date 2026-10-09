---
title: "Visual-Inertial AVL — Trajectory Mode Design"
subtitle: "VPR + simulated IMU + error-state Kalman fusion for GPS-denied UAV navigation"
author: "AVL project"
date: "2026-09-01"
---

# Visual-Inertial AVL — Trajectory Mode Design

> **Status:** implemented (Phases 0–5, 2026-09-01). This doc is both the spec and
> the build log — each section marks what shipped; §15 is the phase table. The
> estimation core is `avl/nav/` (37 unit tests), the runner is
> `scripts/visloc_traj.py`, and the GUI has "Trajectory" + "Studio" tabs.
> Deferred: real-IMU input, other filters (interface in §11), and the small UI
> niceties listed in §13.
>
> **Locked decisions** (from review, 2026-09-01):
> 1. IMU is **synthesised from ground-truth trajectory** with selectable grade presets.
> 2. "Trajectory Studio" draw tool = **snap the drawn path to the nearest real query
>    frames, in order**, and localise that ordered subset.
> 3. Fusion filter = **error-state EKF (15-state)** only. Other filters
>    (complementary, UKF, particle) are out of scope but the module exposes an
>    interface for them (§11).

---

## 1. Motivation

The deployed pipeline is **memoryless**: [`AVLLocalizer.localize`](../avl/localizer.py)
and [`scripts/visloc_eval.py`](../scripts/visloc_eval.py) localise every frame
independently — encode, FAISS cosine retrieval, dedup, top-K, geo-fusion of the top-5
into one lat/lon. [`docs/AVL_Coverage_and_RealTime_Report.md`](AVL_Coverage_and_RealTime_Report.md)
§9.4 already lists a **temporal filter fusing fixes with an IMU** as *missing —
required*, and §8.2 pegs consumer-grade MEMS drift at ~1 % of distance travelled.

Consequences of the memoryless design on a real flight:

| Failure mode | Cause | What trajectory mode adds |
|---|---|---|
| Pose jumps frame-to-frame, unusable by a controller | no motion model | INS propagates pose at 100 Hz between fixes |
| A single stray top-5 tile drags the fix 100s of m off | wide-area retrieval ambiguity | χ² innovation gate rejects fixes inconsistent with the INS prediction |
| Total loss over water / cloud / featureless terrain | retrieval returns garbage or nothing | INS coasts through the dropout; error stays bounded for tens of seconds |
| No velocity or attitude output | not estimated | filter state carries `v`, `q`, and IMU biases |

The UAV-VisLoc region CSVs (`data/UAVVisLoc/<r>/<r>.csv`) are **already time-ordered
trajectories**: `num, filename, date (ISO, ~3 s spacing), lat, lon, height,
Omega, Kappa, Phi1 (yaw), Phi2`. Region 10 = 144 frames ≈ 7 min ≈ 18 m/s over a
~1.5 × 1.4 km mosaic. That gives ground-truth position + attitude + timestamps per
frame — enough to synthesise inertial data and to score a fused trajectory.

---

## 2. Scope

**In scope**

- A **loosely-coupled** visual-inertial estimator: the existing VPR block emits one
  fused pose per frame; that pose is a position measurement to an error-state EKF
  driven by a synthetic IMU.
- Synthetic IMU with grade presets (consumer MEMS / tactical / custom).
- Simulated fix dropouts and decimation, to exercise the coasting behaviour.
- Two new GUI tabs: **Trajectory (VPR + IMU)** (batch run + plots) and
  **Trajectory Studio** (draw a path over the satellite mosaic → snap to real
  frames → run → overlay).
- Three-way scoring: **IMU-only**, **VPR-only**, **Fused**, versus ground truth.

**Out of scope (this pass)**

- Tightly-coupled fusion (feeding raw top-K retrieval candidates into the filter).
- Real recorded IMU logs (interface noted in §5.6 for later).
- Filters other than the error-state EKF (interface in §11).
- Magnetometer / barometer / zero-velocity updates (hooks noted, not built).
- On-vehicle / MAVLink integration.

---

## 3. System pipeline

```
                    ┌──────────────────────── per IMU tick, f_imu = 100 Hz ────────────────────────┐
 GT trajectory      │                                                                              │
 (lat,lon,h,att,t)  │   IMU sim  ─►  ω̃_b(t), f̃_b(t)   ─►   strapdown INS mechanization            │
   spline-fit ──────┤   §5           (bias + rand.walk +      §6   x_ins(t) = [ p, v, q ]           │
                    │                 scale + noise + g)            high-rate dead-reckoned pose    │
                    └───────────────────────────────────────────────────────┬──────────────────────┘
                                                                            │  (also run standalone
                                                                            │   ⇒ "IMU-only" trace)
                                                                            ▼
 query frame k ─► VPREncoder (4 rot) ─► FAISS cosine top-K ─► dedup ─► geo-fusion ─► z_vpr,k = (p_k, R_k)
 (avl/retrieval.py + avl/geo.py, UNCHANGED)                          §4.1        §8   R_k ← confidence, spread
                                                                            │
                        x0 known:                                           ▼
                     p0 = (lat0,lon0,h0)         ┌──────────────  Error-State EKF (15)  ──────────────┐
                     v0 = GT diff | 0     ─────► │  predict:  δx cov ← IMU noise model (F, Q)          │
                     ψ0 = Phi1[0]               │  update (fix arrives, passes χ² gate):              │──► fused
                     P0 = diag(...)             │     r = z_vpr,k − h(x_ins);  K = PHᵀS⁻¹             │    x̂(t)
                                                │     inject δx into nominal; feed back b_a, b_g      │    + P(t)
                                                │  states: δp δv δθ δb_a δb_g  (§7)                   │
                                                └────────────────────────────────────────────────────┘
                                                                            │
                                                                            ▼
        Score 3 trajectories vs GT (§9):  IMU-only  ·  VPR-only  ·  Fused
        horiz RMSE / median / p95 / max · along-track · cross-track · CEP50/95
        · unaided-INS drift-rate (% of path) · fix availability · fixes gated · final error
```

The VPR block is reused **verbatim** — trajectory mode changes nothing about
encoding, retrieval, dedup, or geo-fusion. It only consumes the per-frame fused
pose (and its `confidence` / `spread_m`) that the block already produces.

---

## 4. Coordinate frames and notation

| Symbol | Meaning |
|---|---|
| `n` | navigation frame: local **ENU** tangent plane, origin at `x0` (the known start) |
| `b` | body frame: IMU / camera, x-forward, y-left, z-up |
| `p ∈ ℝ³` | position in `n`, metres `(east, north, up)` |
| `v ∈ ℝ³` | velocity in `n`, m/s |
| `q` | unit quaternion, body→nav rotation; `R{q} ∈ SO(3)` its matrix |
| `f_b` | specific force (accelerometer ideal output), body frame |
| `ω_b` | angular rate (gyro ideal output), body frame |
| `b_a, b_g` | accelerometer / gyro bias, body frame |
| `g_n` | gravity vector in `n` = `(0, 0, −9.80665)` (ENU: down is −up) |
| `[a]_×` | skew-symmetric matrix of `a` |
| `Δt` | IMU sample period = `1 / f_imu` |

**Flat-earth justification.** UAV-VisLoc sorties are < 10 km across. Using a single
ENU tangent plane fixed at `x0` (rather than re-linearising per step) incurs
< 0.1 m of projection error over that span — negligible against the target 100 m
band. Geo ↔ local conversions reuse the WGS-84 metres-per-degree series already in
[`avl/geo.py`](../avl/geo.py) (`_meters_per_degree`, `_to_local_xy`,
`_from_local_xy`).

### 4.1 VPR measurement

For frame `k` at time `t_k`, the VPR block returns fused `(lat_k, lon_k[, alt_k])`,
plus `confidence_k` and `spread_m,k`. Converted to the local frame:

```
z_vpr,k = to_local_xy(lat_k, lon_k) [, alt_k]      # 2- or 3-vector
```

Measurement-noise mapping `R_k` in §8.

---

## 5. IMU simulation

Synthetic only. Goal: from the sparse GT trajectory produce a continuous 100 Hz
stream of `(ω̃_b, f̃_b)` whose **error characteristics match a real MEMS unit**, so
the INS drifts realistically and the filter has something meaningful to estimate.

### 5.1 Trajectory reconstruction

Inputs per frame: `t_i` (from `date`), `p_i` (GT lat/lon/height → local ENU),
attitude `(roll_i, pitch_i, yaw_i)` from `(Omega, Phi2, Phi1)`.

- **Position:** natural **cubic spline** per ENU axis, `p(t) ∈ C²`. Analytic first
  and second derivatives give `v(t) = ṗ(t)` and `a(t) = p̈(t)` (inertial
  acceleration in `n`).
- **Attitude:** convert each frame's Euler triple to a quaternion `q_i`;
  interpolate with **piecewise SLERP** (optionally squad for C¹). Body angular
  rate from finite-difference of the interpolated quaternion:
  `[ω_b(t)]_× = R{q(t)}ᵀ Ṙ{q(t)}`, evaluated on the IMU grid.

No scipy dependency: cubic-spline coefficients (tridiagonal solve) and SLERP are
~40 lines of numpy each.

> **Caveat surfaced for the reviewer:** UAV-VisLoc's `Omega/Kappa/Phi` are
> photogrammetric image-orientation angles, not a flight INS log. Roll/pitch
> recovered from them are approximate. This mainly affects the *lever-arm* and the
> gravity projection into the body frame; the position spline (the dominant driver
> of `f_b`) is unaffected. Presets can down-weight attitude realism if needed.

### 5.2 Ideal sensor outputs

```
a_n(t)  = p̈(t)                            # from the position spline
f_n(t)  = a_n(t) − g_n                     # specific force in n
f_b(t)  = R{q(t)}ᵀ f_n(t)                  # ideal accelerometer
ω_b(t)  = vee( R{q(t)}ᵀ Ṙ{q(t)} )          # ideal gyro
```

### 5.3 Error model

Per-axis, at each tick:

```
ω̃_b = (I + S_g) ω_b + b_g + n_g            n_g  ~ N(0, σ_g²  · f_imu)     # ARW white noise
f̃_b = (I + S_a) f_b + b_a + n_a            n_a  ~ N(0, σ_a²  · f_imu)     # VRW white noise
ḃ_g = n_bg                                 n_bg ~ N(0, σ_bg² · Δt)        # bias random walk
ḃ_a = n_ba                                 n_ba ~ N(0, σ_ba² · Δt)
```

- `S_g, S_a`: constant scale-factor + misalignment matrices, drawn once per run,
  diagonal ~ `N(0, σ_sf²)`, off-diagonal ~ `N(0, σ_mis²)`.
- `b_g(0), b_a(0)`: drawn once per run from `N(0, σ_b0²)` (turn-on bias).
- Optional constant **lever arm** `r_b` (IMU offset from camera): adds
  `ω̇_b × r_b + ω_b × (ω_b × r_b)` to `f_b`. Default `r_b = 0`.

### 5.4 Grade presets

As implemented in `avl/nav/imu_sim.py` (`PRESETS`). `turn-on bias` is the
**post-alignment residual**, not the raw datasheet zero-rate offset: trajectory
mode starts from a known pose + coarse alignment, so the uncalibrated offset
(≈0.5 °/s for a raw consumer gyro — enough to swamp everything over an 11-min
flight) is assumed already removed. "custom" exposes every field in the GUI.

| Parameter | symbol | Consumer MEMS | Tactical | Units |
|---|---|---:|---:|---|
| Gyro noise density (ARW) | `σ_g` | 0.30 | 0.05 | °/√hr |
| Gyro bias random-walk | `σ_bg` | 100 | 8 | °/hr/√hr |
| Gyro turn-on residual (1σ) | `σ_b0,g` | 0.02 | 0.005 | °/s |
| Gyro scale factor (1σ) | `σ_sf,g` | 0.1 | 0.05 | % |
| Accel noise density (VRW) | `σ_a` | 0.10 | 0.02 | (m/s)/√hr |
| Accel bias random-walk | `σ_ba` | 600 | 50 | µg/√hr |
| Accel turn-on residual (1σ) | `σ_b0,a` | 3 | 0.5 | mg |
| Accel scale factor (1σ) | `σ_sf,a` | 0.2 | 0.05 | % |
| Axis misalignment (1σ) | — | 0.02 | 0.01 | ° |

**Observed unaided behaviour** (`scripts/nav_selfcheck.py`, consumer, region 10,
144 frames / 676 s / 9.2 km, from known `x0`, no fixes): residual gyro bias and
its random walk tilt the platform, coupling gravity into horizontal acceleration;
position error grows super-linearly and reaches **km scale by the end of the
11-min flight** (~9 500 m median, ~69 km final — plot it on a clipped axis). Over
a short coast it is far gentler: **tens of metres for a 30–60 s gap.** Bounding
this is the filter's job (§14 for the aided numbers).

### 5.5 Fix scheduling / failure injection

- `--vpr-every N`: run a VPR fix only every `N`-th trajectory frame; INS coasts
  between.
- `--dropout "mm:ss-mm:ss,..."`: withhold all fixes inside these mission-time
  windows (simulated cloud / water / featureless terrain).
- The VPR block still runs on decimated frames; withheld = the fix is computed but
  not passed to the filter (so "VPR-only" and "Fused" see the same raw fixes and
  the comparison is fair).

### 5.6 Real-IMU hook (future)

`avl/nav/imu_sim.py` returns an `ImuStream` dataclass
(`t[N], gyro[N,3], accel[N,3], frame_time_index[...]`). A future `imu_log.py`
loader producing the same `ImuStream` from a CSV/rosbag drops in with no filter
changes.

---

## 6. INS strapdown mechanization

Standard first-order integration on the IMU grid (`avl/nav/ins.py`):

```
q_{t+1} = q_t ⊗ Δq( (ω̃_b − b_g) · Δt )
R_t     = R{q_t}
a_n     = R_t (f̃_b − b_a) + g_n
v_{t+1} = v_t + a_n · Δt
p_{t+1} = p_t + v_t · Δt + ½ a_n · Δt²
```

Run with the filter's current bias estimates for the **fused** solution; run with
`b_a = b_g = 0` (or fixed turn-on bias, no feedback) and no updates for the
**IMU-only** solution.

---

## 7. Error-state EKF (15 states)

Reference formulation: Solà, *Quaternion kinematics for the error-state Kalman
filter* (2017); Groves, *Principles of GNSS, Inertial, and Multisensor Integrated
Navigation Systems*, 2nd ed. (2013), ch. 14.

### 7.1 State

Nominal state `x = (p, v, q, b_a, b_g)` integrated by the INS (§6).
Error state (the filter's actual state vector, 15-D):

```
δx = [ δp(3)  δv(3)  δθ(3)  δb_a(3)  δb_g(3) ]ᵀ
```

`δθ` is a **local** (body-frame) small-angle attitude error: the true attitude is
`q = q̂ ⊗ Δq(δθ)`. This right-multiplied convention pairs directly with the
mechanization step `q̂_{k+1} = q̂_k ⊗ Δq((ω̃−b̂_g)Δt)` and is what
`avl/nav/eskf.py` implements.

### 7.2 Propagation

Continuous-time error dynamics:

```
δṗ   = δv
δv̇   = −R{q̂} [ f̃_b − b̂_a ]_× δθ  −  R{q̂} δb_a  −  R{q̂} n_a
δθ̇   = −[ ω̃_b − b̂_g ]_× δθ  −  δb_g  −  n_g
δḃ_a = n_ba
δḃ_g = n_bg
```

Discretise: `F = I₁₅ + A·Δt` (A = the Jacobian above), `Q` = `G·diag(σ_a², σ_g²,
σ_ba², σ_bg²)·Gᵀ·Δt`. Covariance predict every IMU tick (or at a decimated
`f_filter`, e.g. 50 Hz, for speed):

```
P⁻ = F P Fᵀ + Q
```

The nominal state is propagated by the INS, so the error-state mean stays 0 during
prediction.

### 7.3 Update (VPR fix, loosely coupled)

Measurement = VPR position in the local frame. With altitude fusion on, `m = 3`;
off, `m = 2`.

```
h(x)   = p̂                           (the INS position at t_k)
H      = [ I_m  0  0  0  0 ]         (m × 15)
r      = z_vpr,k − h(x)              (innovation)
S      = H P⁻ Hᵀ + R_k
```

**χ² innovation gate** (this is the outlier rejection that replaces hand-tuning the
geo-fusion cluster radius):

```
γ = rᵀ S⁻¹ r
if γ > χ²_{m, 0.997}:   reject fix, skip update, log as gated
```

Otherwise:

```
K   = P⁻ Hᵀ S⁻¹
δx  = K r
P⁺  = (I − K H) P⁻ (I − K H)ᵀ + K R_k Kᵀ     (Joseph form)
```

**Injection** into the nominal state, then reset `δx → 0`:

```
p̂  ← p̂ + δp
v̂  ← v̂ + δv
q̂  ← q̂ ⊗ Δq(δθ) ,  renormalise      (local error, right-multiplied)
b̂_a ← b̂_a + δb_a
b̂_g ← b̂_g + δb_g
```

The covariance angle-reset term `G = I − ½[δθ]_×` on the `δθ` block is a
second-order correction and is omitted in Phase 1 (documented in the code).

Bias feedback means subsequent INS propagation uses the corrected biases — this is
what bounds the drift during the next coasting interval.

### 7.4 Initialisation from the known start

| State | "known" init | fallback | `P0` (1σ) |
|---|---|---|---|
| `p0` | `x0` → local origin ⇒ `0` | — | 2 m horiz, 3 m vert |
| `v0` | GT finite difference of frames 0–1 | `0` | 0.5 m/s (known) / 5 m/s (fallback) |
| `q0` | yaw = `Phi1[0]`; roll/pitch from `Omega/Phi2` | yaw only, level | 2° yaw / 5° roll-pitch |
| `b_a0` | `0` | `0` | preset turn-on `σ_b0,a` |
| `b_g0` | `0` | `0` | preset turn-on `σ_b0,g` |

`--init-vel {known,zero}` selects the velocity row. Everything else is automatic.

---

## 8. VPR measurement-noise mapping

The filter needs `R_k`, not a scalar confidence. Mapping (tunable, lives in
`avl/nav/eskf.py` config):

```
σ_h,k = σ_floor + α · spread_m,k + β · (1 − confidence_k)          # horizontal, metres
σ_z,k = γ_z · σ_h,k                                               # vertical (if fused)
R_k   = diag(σ_h,k², σ_h,k², σ_z,k²)          # or 2×2 without altitude
```

**Calibrated defaults** (`avl/nav/eskf.py`): `σ_floor = 45 m`, `α = 0.05`,
`β = 35 m`, `γ_z = 2`, gate at `p = 0.99`.

Calibration was done by regressing per-frame `err_vpr_m` against `spread_m` and
`confidence` on `anyloc-lite` / r05 and `denseuav-vit` / r10 residuals
(`scripts/visloc_traj.py` output). Finding: for these encoders the **within-inlier
error is close to homoscedastic** — neither `spread_m` nor `confidence` predicts it
(|corr| < 0.2), and `spread_m` does not separate the gross outliers either
(`confidence` is also on a compressed 0.1–0.55 scale for AnyLoc). So the floor
carries most of the weight and the **χ² innovation gate does the outlier
rejection** (it caught 7 of 9 `>200 m` fixes on the r05 validation run). `α, β`
are kept small as a mild lever for encoders whose scores *are* better calibrated.
Re-tune per encoder from the GUI once Phase 3 lands; report NEES so mis-tuning is
visible.

---

## 9. Metrics (`avl/nav/metrics.py`)

Computed for each of IMU-only / VPR-only / Fused, sampling the estimated trajectory
at the GT frame times:

| Metric | Definition |
|---|---|
| horiz error `e_k` | `‖p̂_k − p_k‖` over ENU east/north |
| RMSE / median / p95 / max / mean | of `{e_k}` |
| CEP50 / CEP95 | 50th / 95th percentile of radial horizontal error |
| along-track / cross-track | `e_k` projected on / perpendicular to GT heading at `k` |
| vertical error | `|û_k − u_k|` (if altitude fused) |
| **unaided INS drift rate** | `e_final(IMU-only) / path_length × 100 %` |
| fix availability | `n_fixes_accepted / n_frames` |
| fixes gated | count rejected by the χ² gate |
| final-position error | `e_{N-1}` |
| filter consistency (NEES) | mean `γ` over accepted fixes (should ≈ `m`) |

Same `{mean, median, p95, min, max}` block shape as
[`visloc_eval.py`](../scripts/visloc_eval.py) `stats()` so the GUI reuses existing
table rendering.

---

## 10. New module: `avl/nav/`

Pure **numpy**, no torch / faiss / Qt — so it is unit-testable in isolation and the
GUI could even call it in-process later.

| File | Contents |
|---|---|
| `avl/nav/__init__.py` | public exports |
| `avl/nav/frames.py` | ENU tangent-plane geo↔local (wraps `avl.geo`), gravity, `skew`, quaternion utils (`exp`, `mul`, `to_R`, `from_euler`, `slerp`) |
| `avl/nav/imu_sim.py` | `GradePreset`, `PRESETS`, `TrajectorySpline`, `simulate_imu(gt, cfg) -> ImuStream` |
| `avl/nav/ins.py` | `mechanize(imu, x0, biases) -> StateTrajectory` |
| `avl/nav/eskf.py` | `ErrorStateEKF` (`predict`, `update`, `_gate`, `_inject`), `EskfConfig` |
| `avl/nav/filters.py` | `Filter` ABC + registry (`"eskf"` only for now) — extension point (§11) |
| `avl/nav/metrics.py` | `trajectory_metrics(gt, est, path_len, gated, ...)` |
| `avl/nav/raster.py` | `RegionRaster` (load `satelliteNN.tif` + corner coords; `geo_to_px` / `px_to_geo` / `thumbnail(max_px)` / `discover` / `for_region`) and `snap_path_to_frames(path_xy, frame_xy, ids, corridor_m)` — numpy + Pillow, Qt-free |
| `tests/test_nav_*.py` | see §14 |

---

## 11. Filter interface (extension point)

Even though only the ES-EKF is built now, `filters.py` defines:

```python
class Filter(Protocol):
    def init(self, x0: NominalState, P0: np.ndarray, cfg) -> None: ...
    def predict(self, imu_tick: ImuSample) -> None: ...
    def update(self, z: np.ndarray, R: np.ndarray) -> UpdateInfo: ...   # UpdateInfo.gated: bool
    def state(self) -> NominalState: ...

FILTERS = {"eskf": ErrorStateEKF}   # future: "complementary", "ukf", "pf"
```

The GUI filter dropdown is populated from `FILTERS.keys()`, so adding a filter is a
one-line registry change plus the class.

---

## 12. New script: `scripts/visloc_traj.py`

A subprocess sibling of `visloc_eval.py` / `visloc_query.py`. Imports torch (for
VPR) **and** `avl.nav` (numpy). The GUI never imports either — it shells out and
renders the JSON, exactly as today.

### 12.1 CLI

```
--refs PATH                 reference CSV (as visloc_eval)
--queries PATH              query CSV; rows are the time-ordered trajectory
--model NAME                encoder (denseuav-vit default)
--fusion {cluster,softmax,median,weighted_mean,top1}   geo-fusion for the VPR block
--frame-stride N            use every N-th row of the query CSV as a trajectory frame
--frame-subset PATH         optional: explicit ordered list of image_ids (Studio uses this)
--vpr-every N               pass a VPR fix to the filter every N-th frame (INS coasts otherwise)
--dropout "s0-s1,s2-s3"     mission-time windows (seconds) with fixes withheld
--imu-rate HZ               default 100
--imu-grade {consumer,tactical,custom}
--imu-<param> VALUE         per-field overrides when --imu-grade custom (see §5.4)
--filter eskf               (only value for now)
--fuse-altitude / --no-fuse-altitude     default: off (horizontal only)
--init-vel {known,zero}     default: known
--seed INT                  RNG seed for the IMU error draw (reproducible runs)
--tag NAME
--out-dir artifacts/visloc
--ref-cache PATH            reuse reference descriptors (as visloc_eval)
```

### 12.2 Retrieval reuse

The shared core already exists: **`avl.retrieval.localize()`** (encode N rotations →
cosine vs refs → per-location dedup → top-K → `weighted_geo_fusion`) is what
`scripts/visloc_query.py` uses. `visloc_traj.py` calls it once per frame with
`ground_truth=(lat, lon)` so `confidence`, `spread_m`, and `fused_error_m` come
back filled in. `scripts/visloc_eval.py` still carries its own equivalent copy of
that loop; it was **left untouched** (no benchmark-behaviour risk) — migrating it
onto `avl.retrieval.localize` is optional future cleanup, not part of this work.

### 12.3 Outputs (in `artifacts/visloc/`)

**`<tag>_traj.csv`** — one row per trajectory frame:

| Column | Meaning |
|---|---|
| `frame`, `t_s`, `query_id` | index, mission time, image id |
| `gt_lat`, `gt_lon`, `gt_alt` | ground truth |
| `ins_lat`, `ins_lon`, `ins_alt` | IMU-only mechanization |
| `vpr_lat`, `vpr_lon` | raw VPR fix (`nan` if frame not a fix frame) |
| `vpr_conf`, `vpr_spread_m` | from the VPR block |
| `fused_lat`, `fused_lon`, `fused_alt` | ES-EKF output |
| `err_ins_m`, `err_vpr_m`, `err_fused_m` | horizontal error vs GT |
| `along_m`, `cross_m` | fused along/cross-track error |
| `fix_used` (0/1), `fix_gated` (0/1) | filter's disposition of this frame's fix |
| `P_pos_sigma_m` | `√trace(P[0:2,0:2])` — filter's own horizontal 1σ |
| `vx`, `vy`, `vz` | fused velocity, ENU |
| `bias_a_norm`, `bias_g_norm` | estimated bias magnitudes |

**`<tag>_traj.json`** — config echo + summary + polylines for plotting:

```jsonc
{
  "tag": "r10__denseuav-vit__eskf",
  "model": "denseuav-vit", "fusion": "cluster", "filter": "eskf",
  "imu": { "grade": "consumer", "rate_hz": 100, "seed": 0, "params": { ... } },
  "vpr_every": 1, "frame_stride": 1, "fuse_altitude": false, "init_vel": "known",
  "dropouts_s": [[120, 180]],
  "n_frames": 144, "duration_s": 431.0, "path_length_m": 6980.0,
  "origin": { "lat": 40.3505, "lon": 115.7770, "alt": 772.4 },
  "region_mosaic": { "path": "data/UAVVisLoc/10/satellite10.tif",
                     "lt": [40.355093, 115.776356], "rb": [40.341475, 115.794041] },

  // each metric block = { mean, median, p95, min, max } unless noted
  "ins_only":  { "horiz_error_m": {...}, "final_error_m": 152.4, "drift_rate_pct": 2.18 },
  "vpr_only":  { "horiz_error_m": {...}, "cep50_m": 41.0, "cep95_m": 210.6,
                 "final_error_m": 63.2, "n_outliers_gt_200m": 7 },
  "fused":     { "horiz_error_m": {...}, "along_track_m": {...}, "cross_track_m": {...},
                 "cep50_m": 22.7, "cep95_m": 58.9, "final_error_m": 18.4,
                 "fix_availability": 0.86, "fixes_gated": 12, "mean_nees": 2.9 },

  "series": {
    "t_s":        [ ... ],
    "gt_en":      [[e,n], ...],
    "ins_en":     [[e,n], ...],
    "vpr_en":     [[e,n]|null, ...],
    "fused_en":   [[e,n], ...],
    "err_ins_m":  [ ... ], "err_vpr_m": [ ... ], "err_fused_m": [ ... ],
    "P_sigma_m":  [ ... ],
    "fix_used":   [ 0|1, ... ], "fix_gated": [ 0|1, ... ]
  }
}
```

`series.*_en` are in local ENU metres (origin = start) so the GUI plots them with a
trivial auto-fit; `origin` + `region_mosaic` let Studio convert back to pixels.

---

## 13. GUI changes

Both tabs follow the existing thin-shell pattern
([`avl/model_bench_gui.py`](../avl/model_bench_gui.py)): a `QProcess` runs
`visloc_traj.py`, stdout streams to the log pane, on exit the JSON is loaded and
rendered. Reuses `_make_process`, `_spawn`, `_read_output`, `_run_finished`,
`set_status`, the progress strip, and the theme tokens from
[`model_bench_ui_spec.md`](model_bench_ui_spec.md) §10.

Tabs become: `Pipeline | Comparison | Localize | Trajectory | Studio`
(`_build_results`, [model_bench_gui.py:924](../avl/model_bench_gui.py#L924)).

### 13.1 "Inertial fusion" control group — **as built**

A `QGroupBox` inside the Trajectory tab ([`_build_trajectory_tab`](../avl/model_bench_gui.py)):

| Control | Type | Values / default |
|---|---|---|
| IMU error model | combo | Consumer MEMS *(default)* / Tactical grade / **none — perfect IMU** (the "IMU off" case: `--imu-grade perfect`, no inertial error, so the filter+gate are isolated) |
| IMU noise × | double-spin | 0.1–10.0, default 1.0 — scales the preset (`--imu-scale`), the CLI stand-in for the "custom" sliders |
| Fusion filter | combo | Error-State EKF (15-state) *(only entry; from `TRAJ_FILTER_CHOICES`)* |
| VPR fix rate | spin | every `N` frames, 1–20, default 1 |
| Fix dropouts | line edit | seconds, e.g. `120-180, 300-330`, default empty |
| Initial velocity | combo | Known (from GT) *(default)* / Zero |
| Fuse altitude | checkbox | default off |
| Real frame timestamps | checkbox | default on — sources per-frame time from the matching `data/UAVVisLoc/<r>/<r>.csv`, else uniform 3 s |
| Frame stride / Max frames | spin / spin | 1 / 120 |
| RNG seed | spin | default 0 |

Reference database, query set, active encoder, geo-fusion, rotations, and crop are
inherited from the existing left panel — not duplicated.

### 13.2 Tab "Trajectory (VPR + IMU)"

**Purpose:** run the full estimator on a whole prepared query set, read the
three-way comparison.

Layout **as built**:

```
┌ Trajectory · VPR + IMU  ────────────────────────────────────────────┐
┌ card: [Inertial fusion QGroupBox — §13.1]  ·  "uses left panel …"   │
│       [Run trajectory]   <hint: N frames · Xs · path · ms/frame …>  │
└────────────────────────────────────────────────────────────────────┘
┌ ground track  (TrajectoryPlot, equal-aspect) ──────────────────────┐
│  ── ground truth (grey)   ── IMU-only (red)                        │
│  •  VPR fixes (amber)      ── fused (blurple)                      │
└────────────────────────────────────────────────────────────────────┘
┌ horizontal error vs time  (TrajectoryPlot; dropout windows shaded) ┐
│  IMU-only clipped to 4× the VPR/fused max so the shape stays legible│
└────────────────────────────────────────────────────────────────────┘
┌ metrics (3×9) ────────────────────────────────────────────────────┐
│ Track      median  p95   max   RMSE  final  drift%  fix avail  gated │
│ IMU-only / VPR-only / Fused        (p95 & final colour-coded)       │
└────────────────────────────────────────────────────────────────────┘
     …subprocess stdout still streams to the shared log on the Comparison tab.
```

Plots are a **custom `TrajectoryPlot(QWidget)` with `paintEvent`** (polylines,
gridlines, tick labels, legend, shaded x-bands, optional equal-aspect) — ~130
lines, no matplotlib/pyqtgraph (neither installed; keeps the frozen app small).
The same widget draws the ground track and the error-vs-time chart. Metric cells
for `p95` / `final` are coloured green < 50 m, amber < 150 m, red otherwise
(mirrors the Comparison `<100 m` rule).

Subprocess wiring mirrors the single-image flow: `self.traj_process`,
`run_trajectory` → `_traj_args` → `_spawn`; `_read_traj_output` parses the
`N/M frames localized` progress line; `_traj_finished` loads `<tag>_traj.json` and
`_render_trajectory` fills the two plots + the table. `_busy`, `stop_benchmark`
and `closeEvent` were extended to cover `traj_process`.

Not built (deferred): a time scrubber under the plots that highlights the frame and shows
its row from `<tag>_traj.csv`.

### 13.3 Tab "Trajectory Studio" — synthetic-mission mode

**Purpose:** the drawn route **is** the flight path. Simulate the IMU from it, and
synthesise VPR fixes as `GT + error` where the error magnitude is drawn from a real
benchmark's per-frame distribution for this region + encoder. The fused track then
follows the drawn route, which is what "draw a route and see the estimator run"
should mean. (An earlier design snapped the route to real drone frames; that only
worked when the route retraced the actual flight, and it is dropped.)

Layout **as built** ([`_build_studio_tab`](../avl/model_bench_gui.py)):

```
┌ Trajectory Studio ─────────────────────────────────────────────────┐
┌ bar: "region NN · W×H px"   speed [20 m/s]  fix every [3.0 s]      │
│      [Draw route]  [Clear]  [Run mission]                          │
└────────────────────────────────────────────────────────────────────┘
┌ MosaicView (QGraphicsView, pan/zoom; auto-fits to the run, dbl-click resets) ┐
│   left-click in Draw mode lays amber dashed vertices               │
│   after Run  : GT (= route, grey) / IMU-only (red) / fused (blurple) lines,  │
│                VPR fixes (cyan dots) overlaid on the mosaic        │
└────────────────────────────────────────────────────────────────────┘
 hint: N frames · X km · VPR-only median A m → Fused median B m (RMSE, final) · IMU drift % · gated
```

- **Region** is inferred from the left panel's query set (`rNN_t*` → `NN`).
  **IMU grade / scale, filter, VPR rate, dropouts, seed, init-vel are read from the
  Trajectory tab.**
- **Needs the benchmark CSV** `artifacts/visloc/<db_tag>__<model>_per_query.csv`
  (from a Comparison-tab run of that encoder on that region). Missing → `Run
  mission` is disabled with a message pointing at the Comparison tab.
- **`RegionRaster`** ([avl/nav/raster.py](../avl/nav/raster.py), numpy + Pillow,
  Qt-free): corner lat/lons from `satellite_ coordinates_range.csv`, `geo_to_px` /
  `px_to_geo`, `thumbnail(max_px=1600) → (PIL.Image, scale)`. Scene coords are
  thumbnail pixels; overlays convert `geo → full px → × scale`.
- **`Run mission`** writes `{waypoints, speed_mps, alt_m, fix_dt_s}` JSON and
  launches `visloc_traj.py --synthetic-path … --vpr-error-csv …` — **no retrieval,
  no torch** (`avl.retrieval` is imported lazily only for the real-frame path).
  `synth_mission()`:
  - **`_round_corners()`** fine-resamples the polyline and box-smooths it to a
    minimum turn radius `r_min = speed² / max_lat_accel` (4 m/s²) — a hand-drawn
    kink is otherwise an acceleration spike once the IMU sim differentiates the
    path twice, and the INS runs off the map.
  - resamples the rounded curve at `speed·fix_dt` for GT, `simulate_imu` from that,
    and `z_vpr = gt + mag·(cosθ, sinθ)` with `mag`, `confidence`, `spread_m`
    sampled from the CSV columns.
  - returns `sigma_floor_m = clip(median(mag pool), 20, 400)`; `main()` sets the
    filter's `sigma_floor_m` to it (and `alpha_spread = beta_conf = 0`) for
    synthetic runs. The §8 calibrated 45 m floor assumes a tighter encoder;
    against a 120 m-noisy pool it over-trusts each fix, corrupts velocity, and the
    INS diverges. Matched R keeps the fused solution locked even when IMU-only
    drifts to kilometres.
- `visloc_traj.py` `main()` is split into `synth_mission()` / `real_mission()`
  feeding the shared fusion + output; the JSON gains `"mode": "synthetic"|"real"`.
- Shares `self.traj_process` with the Trajectory tab (`self._traj_dest` routes the
  finish handler); `_busy` blocks concurrent runs.
- `contiguous_run_along_path` / `snap_path_to_frames` stay in `raster.py` (tested)
  for a possible future real-frame Studio mode; the GUI no longer calls them.

**Not built (deferred):** hover tooltips on the fix dots; an "Export mission" JSON.

### 13.4 Data-contract additions to `model_bench_ui_spec.md`

Done — [`model_bench_ui_spec.md`](model_bench_ui_spec.md) §12 documents the two
tabs, the "Inertial fusion" controls ↔ `visloc_traj.py` flags, the CSV/JSON
contracts, the `TrajectoryPlot` / `MosaicView` widgets, and the snap algorithm.
The runtime-state table (§9) gained `TRAJECTORY <tag>` / `STUDIO <tag>`.

---

## 14. Validation

**Unit tests — `tests/test_nav_*.py`, stdlib `unittest`, numpy only, no GPU.**
Run: `.venv/bin/python -m unittest discover -s tests -p "test_nav_*.py"`.
**Status: 37 tests, all passing.**

| File | Covers |
|---|---|
| `test_nav_frames` | skew=cross; Euler↔quat and rot↔quat round-trips; exp/log map (|φ|<π); `quat_mul` composes rotations; `slerp` endpoints/midpoint; geo→ENU→geo < 1e-6 m over region 10; ENU axis signs |
| `test_nav_ins` | stationary level body stays put; constant east specific force ⇒ ½at²; constant yaw rate ⇒ linear heading; measured rate == bias ⇒ no rotation |
| `test_nav_imu_sim` | cubic spline interpolates knots exactly and tracks a cubic in the interior; **perfect** preset ⇒ INS reproduces the GT spline < 1 m over 60 s; noise-free ⇒ `gyro==true_gyro`; stationary level ⇒ accel reads +g; consumer drifts, tactical drifts less than consumer |
| `test_nav_eskf` | χ² table values; 500 m outlier ⇒ `gated`, state unchanged; large-P₀ + one good fix pulls the state; P shrinks on update; **fusion bounds the drift** (fused final < 0.5 × INS-only, p95 < 25 m, median < 12 m); mid-flight outlier gated and solution unperturbed; 30 s dropout ⇒ error < 80 m and < 0.25 × unaided INS over the gap |
| `test_nav_metrics` | horizontal error ignores altitude; cross-track on an eastbound leg; straight-line path length; drift-rate / fix-availability bookkeeping |
| `test_nav_raster` | region-05 corners ↔ pixel bounds; geo↔px round-trip; ENU axis orientation; `snap_path_to_frames` orders by arc-length and respects the corridor |

**Integration self-check — `scripts/nav_selfcheck.py`** (no encoder; VPR fixes are
GT + Gaussian noise). Observed on `data/UAVVisLoc/10/10.csv` (144 frames, 676 s,
9.2 km, 13.7 m/s), consumer IMU, 25 m fix noise:

| Track | median | p95 | max | final | drift |
|---|---:|---:|---:|---:|---:|
| IMU-only (unaided 11 min) | 9 500 m | 63 000 m | 69 000 m | 69 000 m | 747 % |
| VPR-only (raw fixes) | 24 m | 62 m | 99 m | — | — |
| **Fused (ES-EKF)** | **12 m** | **29 m** | **40 m** | **10 m** | **0.11 %** |
| Fused, 40 s fix dropout `180–220 s` | 15 m | 42 m | 52 m | 1 m | 0.01 % |

The filter roughly halves the median vs raw VPR and cuts the p95/max tail ~2–3×;
a 40 s outage stays inside the 100 m band and re-converges within one or two
fixes. (Plot the IMU-only track on a clipped or log axis — it leaves the frame.)

**End-to-end with real retrieval — `scripts/visloc_traj.py`.** Region 05
(`r05_t250`, `anyloc-lite`, `cluster` fusion), first 50 frames (153 s, 3.0 km),
consumer IMU, a fix every frame, cached reference descriptors:

| Track | median | p95 | max | <100 m | <200 m |
|---|---:|---:|---:|---:|---:|
| IMU-only (unaided) | 113 m | 1 536 m | 1 775 m | — | — |
| VPR-only (raw per-frame fix) | 70 m | 652 m | 810 m | ~64 % | ~74 % |
| **Fused (ES-EKF)** | **68 m** | **308 m** | **403 m** | **74 %** | **84 %** |

The χ² gate rejected **7 of the 9 `>200 m` VPR outliers**; median is preserved
(fusion can't beat the encoder's ~70 m inherent error, only better retrieval can),
while p95/max are roughly halved and `<100 m` recall rises 10 points. Region 10
retrieval is weaker for every encoder on file (fused median ~300 m, `<100 m`
~8–12 %), so r05 is the clearer demonstrator; the pipeline runs on either.

**Still to validate in later phases:** `R_k` re-tuning per encoder + NEES check
from the GUI (Phase 3); Studio snap contiguity (Phase 4).

**Figures for the write-up:** (a) three trajectories on the region-10 mosaic;
(b) error-vs-time with the dropout band shaded, IMU-only vs Fused;
(c) estimated gyro-bias converging; (d) NEES vs the χ² envelope.

---

## 15. Phased implementation plan

| Phase | Deliverable | Files | Depends on |
|---|---|---|---|
| **0 ✅** | This document | `docs/VisualInertial_AVL_Design.md` | — |
| **1 ✅** | `avl/nav/` core + unit tests (32 at Phase 1, 37 with raster) + `scripts/nav_selfcheck.py`, validated on synthetic + region-10 GT (fake fixes = GT + noise) | `avl/nav/{__init__,frames,imu_sim,ins,eskf,filters,metrics}.py`, `tests/test_nav_*.py`, `tests/nav_helpers.py`, `scripts/nav_selfcheck.py` | 0 |
| **2 ✅** | `scripts/visloc_traj.py` on `avl.retrieval.localize` (already shared; `visloc_eval.py` left untouched); CSV + JSON contracts (§12.3); calibrated `EskfConfig` defaults; validated end-to-end on r05 + r10 | `scripts/visloc_traj.py`, `avl/nav/eskf.py` (calibrated defaults) | 1 |
| **3 ✅** | Tab "Trajectory (VPR + IMU)": `TrajectoryPlot` paintEvent widget, "Inertial fusion" control group, `traj_process` subprocess wiring, ground-track + error-vs-time plots, 3×9 metrics table; validated headless end-to-end | `avl/model_bench_gui.py` | 2 |
| **4 ✅** | Tab "Trajectory Studio": `avl/nav/raster.py` (`RegionRaster` + `snap_path_to_frames`, 5 tests), `MosaicView(QGraphicsView)`, draw → snap → `--frame-subset` run → mosaic overlay; `visloc_traj.py` uniform-time fallback for re-ordered subsets; validated headless end-to-end. Export-run JSON deferred. | `avl/nav/raster.py`, `avl/model_bench_gui.py`, `scripts/visloc_traj.py` | 3 |
| **5 ✅** | Docs: `model_bench_ui_spec.md` §12 (tabs, controls↔flags, CSV/JSON contracts, widgets, snap), README "Trajectory mode" feature + tab rows + project layout, coverage-report §9.2/§9.3 updated from "missing — required" to "prototyped offline". `requirements.txt` unchanged (spline / SLERP / plots / χ² table all hand-rolled). | `docs/model_bench_ui_spec.md`, `README.md`, `docs/AVL_Coverage_and_RealTime_Report.md` | 4 |

Phases 1–2 are the estimation substance and can be reviewed from the CLI before any
GUI work. Phases 3–4 are additive to the GUI — no existing tab changes behaviour.

---

## 16. Risks / open points

| Risk | Mitigation |
|---|---|
| UAV-VisLoc attitude angles are photogrammetric, not a flight log ⇒ synthetic gyro signal is approximate | position spline (dominant `f_b` driver) is solid; expose an "attitude realism" weight; document the caveat in outputs |
| `R_k` mapping from `confidence`/`spread` mis-scaled ⇒ filter over/under-trusts VPR | §8 calibration step against region-10 residuals; report NEES so mis-tuning is visible |
| 100 Hz × ~7 min × 15×15 covariance in pure Python may be slow | decimate covariance predict to `f_filter = 50 Hz`; vectorise; it is still a one-off offline run, not real-time |
| Large `.tif` mosaics in the Qt view | `RegionRaster.thumbnail(max_px)` downsample; keep full-res only for geo math |
| No matplotlib/pyqtgraph installed | custom `paintEvent` plot widget (≈150 lines), shared by both tabs; keeps the frozen app small |
| Frozen-app (`pyinstaller`) must still bundle only the thin shell | `visloc_traj.py` is a subprocess like the others; `avl.nav` is numpy-only and already inside the package |

---

## 17. References

1. Solà, J. *Quaternion kinematics for the error-state Kalman filter.* arXiv:1711.02508, 2017.
2. Groves, P. D. *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed. Artech House, 2013 (ch. 5, 14).
3. Titterton, D. & Weston, J. *Strapdown Inertial Navigation Technology*, 2nd ed. IET, 2004.
4. Qin, T., Li, P., Shen, S. *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator.* IEEE T-RO, 2018 (loosely/tightly-coupled VI fusion context).
5. Farrell, J. A. *Aided Navigation: GPS with High Rate Sensors.* McGraw-Hill, 2008.
6. Dai, M. et al. *UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization.* 2024 (the trajectory + satellite data used here).
7. PX4 `ecl/EKF2` and ArduPilot `EKF3` external-vision position interfaces (target consumer of the fused pose stream).
8. Project internal: [`docs/AVL_Coverage_and_RealTime_Report.md`](AVL_Coverage_and_RealTime_Report.md) §8–9 (real-time requirements, temporal-filter gap); [`docs/model_bench_ui_spec.md`](model_bench_ui_spec.md) (GUI architecture, data contracts).
