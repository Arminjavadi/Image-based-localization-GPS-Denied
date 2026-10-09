# VO + AVL fusion: offline experiment

*2026-10-09. Code: `avl/nav/vo_fusion.py`, `scripts/vo_fusion_experiment.py`,
tests `tests/test_nav_vo_fusion.py`. Raw results: `artifacts/visloc/vo_fusion/`.*

## Question

Does fusing visual(-inertial) odometry with AVL raise localization accuracy, especially
where the encoder is weak (areas it never saw) or fails (featureless terrain)? Which fusion
method works best?

## Setup

**The AVL side is real.** For 10 encoder settings on 4 regions (the first 144 frames of each
flight), the full frame × tile similarity matrix is rebuilt from the cached descriptors, using
the shipped recipe's heading pick and causal flight centering. The loader reproduces the known
top-1 within 100 m numbers exactly (r10 DenseUAV 18.8 %, r06 MegaLoc 43.1 %, r05 MegaLoc 74.3 %,
r05 ensemble 82.6 %, r11 AGL 65.3 %). These settings span AVL quality from 19 % to 83 %.

**The VO side is simulated** from the ground-truth track, because UAV-VisLoc has no video and
no IMU. Four VO levels:

| Level | Per-step error | Across the dataset's 33–600 s frame gaps |
|---|---|---|
| VIO ~1 % | 1 % scale, 0.5° heading (bias + noise) | Keeps tracking: a real drone films continuously |
| VIO ~3 % | 3 %, 2° | Keeps tracking |
| sparse VO ~3 % | 3 %, 2° | Lost; IMU coast, error grows with time squared (~130 m after 36 s) |
| poor VO ~8 % | 8 %, 5° | Lost; IMU coast |

**Scenarios:**
- `nominal`: the data as recorded.
- `avl_outage`: for 2 × 20 frames, AVL returns confident garbage (each frame's scores shuffled
  over the map); VO still works.
- `both_outage`: the same blocks with VO lost too, as over water or uniform sand. Only the IMU
  coast remains there.

**Methods** (all causal; frame *k* uses only data up to *k*):

| Method | What it is |
|---|---|
| `avl_top1` / `avl_cluster` | AVL alone: best tile / current top-5 cluster fusion |
| `vo_only` | Dead reckoning from the known start |
| `kf` | Position Kalman filter, χ² gated. This is the current ESKF's logic in 2-D. |
| `kf_reanchor` | `kf` + reset when the last 4 global AVL fixes agree along the VO track but disagree with the state |
| `kf_window` | `kf_reanchor` + search-window prior (AVL searches only within 3σ of the prediction) |
| `pf` | Particle filter, 1000 particles, weighted by the **whole** similarity map |
| `pgo` | Sliding-window (15 frames) pose graph, Cauchy-robust AVL factors, IRLS |
| `align` | FoundLoc-style: RANSAC rigid fit of the last 25 VO poses to the AVL fixes; no known start |
| `pf_kidnapped` | `pf` started uniformly over the map (no known start) |

Each cell is the mean of 5 seeds. "within 100 m" counts frames 1–143.

## Results

### Mean over all 10 settings (AVL alone: 49 % within 100 m, median 180 m)

Within 100 m (%):

| Scenario / VO | AVL | kf | kf_reanchor | kf_window | **pf** | pgo | align | pf_kidnapped |
|---|---|---|---|---|---|---|---|---|
| nominal · VIO ~1 % | 49 | **100** | 88 | 85 | 89 | 96 | 78 | 85 |
| nominal · VIO ~3 % | 49 | 88 | 84 | 84 | 88 | 88 | 75 | 83 |
| nominal · sparse VO ~3 % | 49 | 51 | 66 | 67 | **79** | 63 | 54 | 75 |
| nominal · poor VO ~8 % | 49 | 50 | 62 | 64 | **74** | 62 | 44 | 70 |
| AVL outage · VIO ~1 % | 36 | **99** | 90 | 88 | 88 | **99** | 67 | 81 |
| AVL outage · sparse VO | 36 | 36 | 54 | 53 | **63** | 42 | 44 | 59 |
| both outage · VIO ~1 % | 36 | 35 | 59 | 58 | **67** | 45 | 57 | 60 |
| both outage · sparse VO | 36 | 27 | 49 | 51 | **56** | 38 | 42 | 53 |

Median error, nominal: AVL 180 m. With VIO ~1 %: kf 18 m, pf 17 m, pgo 20 m. With sparse VO:
pf 44 m, kf_window 88 m, plain kf 202 m.

### Per setting: nominal scenario, sparse VO (the honest case for this dataset)

| Setting | AVL | kf | kf_window | pf | pgo |
|---|---|---|---|---|---|
| r10 DenseUAV | 19 | 39 | 40 | **84** | 46 |
| r06 DenseUAV | 28 | 34 | 26 | **59** | 53 |
| r10 MegaLoc | 37 | 65 | 81 | **93** | 79 |
| r06 MegaLoc | 43 | 33 | 59 | **63** | 53 |
| r10 MegaLoc+AnyLoc-L | 48 | 59 | 66 | **92** | 74 |
| r06 MegaLoc+Game4Loc | 48 | 34 | 57 | **60** | 56 |
| r05 AnyLoc-L | 50 | 63 | 88 | **97** | 71 |
| r11 MegaLoc (AGL) | 66 | 52 | **72** | 57 | 54 |
| r05 MegaLoc | 74 | 69 | **91** | 90 | 71 |
| r05 ensemble | 83 | 68 | **92** | **92** | 76 |

### Compute (this laptop's CPU, per frame)

kf variants 0.1–0.2 ms · pgo 3.4 ms · align 15 ms · pf (1000 particles) 30 ms. All of these
are negligible next to the encoder, which takes about 1 s per frame.

## Findings

1. **Fusion raises accuracy a lot, and most for weak encoders in unseen areas.** With sparse VO,
   the particle filter takes r10 DenseUAV from 19 % to 84 % and r10 MegaLoc from 37 % to 93 %.
   With continuous VIO, almost every setting reaches 97–100 %. Motion consistency makes even a
   poor single-frame encoder usable: correct matches agree with each other along the VO track,
   and wrong ones scatter at random.
2. **VO quality matters more than the fusion method.** Continuous VIO, even at 3 % drift, beats
   the best method running on sparse VO. Across gaps the simulated IMU coast costs a lot, so on
   the drone, VO/VIO should run continuously and not only on the frames sent to AVL.
3. **The particle filter is the most robust method when VO is imperfect, and when AVL or both
   sensors fail.** It is best in 9 of 12 scenario × VO rows. It does not have to trust the
   top-1 match, because it scores particles against the whole similarity map.
   Its tail is heavier, though: nominal p95 is 500–1000 m vs 43–66 m for kf/pgo with good VIO.
   When it locks onto a wrong mode, the error is large.
4. **The plain KF (today's ESKF logic) is excellent only with good VIO.** With sparse VO it
   locks: after an IMU coast it accepts one wrong fix, then gates out correct fixes for 40+
   frames (seen on r05). In `both_outage` it collapses (27–35 %), because its own large
   uncertainty lets garbage fixes through. The reanchor rule fixes most of this (51 → 66 %).
   It costs about 10 points when VIO is near-perfect, because a wrong place can look
   consistent for 4 frames (seen on r10).
5. **In closed loop the search-window prior adds little** (kf_window vs kf_reanchor: ±3 points).
   The open-loop sweep in `artifacts/visloc/prior/` was much more optimistic, because it
   centred the window on the truth plus noise. Once the window is centred on the filter's own
   estimate, a wrong estimate hides the right tile. Do not count on it as the main gain.
6. **A known start is not essential.** `pf_kidnapped` gets within 2–6 points of `pf` in the
   nominal and AVL-outage rows. FoundLoc-style `align` is weaker than the filters.
7. **Real featureless case found in the data.** r06 frames 92–143 are a separate take-off
   log, and AVL is right on 0–2 % of them with every encoder. No map-based method can help
   there. Only VIO dead reckoning carries through, and only the plain KF with VIO ~1 % uses it
   fully (100 %); the robust methods get pulled toward the garbage fixes (~64 %). This is the
   limit: fusion cannot create information that neither sensor has.

## Real VO on the dataset frames (added the same day)

`scripts/visloc_vo.py` (module `avl/nav/sparse_vo.py`) measures VO on the same frames:
1. SIFT matching between consecutive frames, with images capped at 1200 px.
2. A RANSAC similarity fit (rotation, scale, shift).
3. Tilt correction: measure from the pixel below the drone, not the image centre.
4. Scale from height above ground (baro − DEM) and the per-region camera constant k.
5. Rotation to east/north with the logged heading.

It takes about 0.5 s per frame on this CPU. Results: `artifacts/visloc/vo_real/`.

- **Tilt convention.** The log has 2–14° camera pitch/roll; at 367 m above ground, 10° moves
  the image centre by 65 m. The convention *Omega → +x (image right), Kappa → +y (image down)*
  was the best of all 8 axis/sign assignments on r05, r06 and r11. It cuts r05's median step
  error from 48 % to 12 %.
- **Heading.** The image rotation between frames matches the logged yaw change to within 1°.

| Region | Pairs matched | Median step error | Length ratio (VO ÷ truth) | Direction scatter (MAD) |
|---|---|---|---|---|
| r05 | 142/143 | 12 % (7 m) | 0.95 | 4° |
| r06 | 133/143 | 20 % (frames ≤ 91), 57 % in the take-off tail | 1.07 | 6° |
| r10 | 143/143 | 34 % → **26 %** with corrected k (see below) | 1.21 at the old k | 9° |
| r11 | 141/143 | 7 % (19 m) | 0.97 | 2° |

The ground-truth GPS has its own noise of a few metres on 60 m steps, so part of these errors
is in the reference, not the VO.

**r10's camera constant was wrong, and VO settles it.** The VO ground step scales *with* k
(metres per pixel = k · AGL / width). At the recipe's k = 1.197, r10's VO steps came out 1.21×
the GPS steps, so k ≈ 1.197 / 1.21 = **0.988**. That is close to r05's 0.971, so the two
3000 px regions look like one lens. The old patch-size scan had left r10 unresolved
(MegaLoc 1.48 vs DenseUAV-ViT 1.20). The pipeline now uses 0.988 (`avl/pipeline.py`).

- **Retrieval is unchanged.** The altitude gate already left every r10 frame uncropped at
  1.197, and a smaller k only shrinks the footprint further.
- **The larger value is ruled out:** measured with the MegaLoc recipe (top-1 within 100 m),
  k = 1.45 drops r10 from 37.5 % to 29.9 % (the gate crops 91 % of frames). A 290 m map sized
  for k = 1.45 scores 31.2 %. The planner's tile size at k = 0.988 (~194 m) matches the 200 m
  map that scores best.
- With k = 0.988, r10's VO step error falls from 34 % to 26 %. The 9° direction scatter remains,
  possibly a stabilised gimbal, since tilt correction does not help on r10.

### Fusion with the real VO

Within 100 m (%), nominal scenario, 5 seeds. Failed VO steps coast on the simulated IMU; the
filters assume VO 1σ = 20 % of the step + 10 m for every region. `hybrid` is defined in the
next section.

| Setting | AVL | kf | kf_reanchor | kf_window | pf | pgo | **hybrid** |
|---|---|---|---|---|---|---|---|
| r10 DenseUAV | 19 | 62 | 62 | 60 | 87 | 50 | **87** |
| r06 DenseUAV | 28 | 48 | 48 | 25 | 60 | 58 | **60** |
| r10 MegaLoc | 37 | 19 | 69 | 74 | 94 | 71 | **94** |
| r06 MegaLoc | 43 | 21 | 51 | 53 | 63 | 46 | **62** |
| r10 MegaLoc+AnyLoc-L | 48 | 86 | 83 | 82 | 95 | 83 | **95** |
| r06 MegaLoc+Game4Loc | 48 | 45 | 56 | 57 | 60 | 60 | **60** |
| r05 AnyLoc-L | 50 | 91 | 91 | 69 | 100 | 95 | **100** |
| r11 MegaLoc (AGL) | 66 | 73 | 85 | 80 | 81 | 77 | **80** |
| r05 MegaLoc | 74 | 94 | 94 | 90 | 100 | 96 | **100** |
| r05 ensemble | 83 | 99 | 96 | 94 | 100 | 99 | **100** |
| **Mean** | **49** | 64 | 73 | 68 | 84 | 74 | **84** |

Mean over the 10 settings:

| Scenario | AVL | kf_reanchor | pf | pgo | **hybrid** |
|---|---|---|---|---|---|
| nominal: within 100 m / median / p95 | 49 % / 180 m / 1356 m | 73 % / 72 m / 401 m | 84 % / 47 m / **5305 m** | 74 % / 69 m / 478 m | **84 % / 47 m / 530 m** |
| AVL outage: within 100 m / median | 36 % / 322 m | 60 % / 157 m | 70 % / 63 m | 56 % / 158 m | **69 % / 67 m** |
| both outage: within 100 m / median | 36 % / 322 m | 51 % / 181 m | 64 % / 95 m | 41 % / 300 m | **62 % / 97 m** |

How the gains split:
- **Good VO** (r05 at 12 %, r11 at 7 %): all r05 encoders reach 100 %, including AnyLoc-L at
  50 % alone.
- **r10 after the k fix:** even the weakest encoder (DenseUAV, 19 %) reaches 87 %.
- **r06:** gains are smallest (+12 to +32 points). Its take-off tail has no usable AVL.

## Hybrid filter and the app (added the same day)

The plain particle filter is the most accurate, but it occasionally fails by kilometres. Its
mean p95 is 5.3 km, from r06 MegaLoc / MegaLoc+Game4Loc, where it reaches 24 km. That is
not acceptable on a vehicle.

`hybrid` (in `avl/nav/vo_fusion.py`) keeps the particle filter as the global layer. Its mode is
the output while it is confident (≥ 60 % of the weight within 200 m of the mode), and a gated
Kalman filter with a search window carries the estimate when it is not. The result keeps the
PF's accuracy (84 % within 100 m, median 47 m) and cuts the mean p95 from 5.3 km to 530 m.

An alternative was tested and rejected: making the Kalman filter the output and only
re-seeding it from a confident, disagreeing PF scored 76 % / 54 m / 573 m with real VO.

With simulated continuous VIO (~3 %), hybrid and pf tie at 88 % within 100 m. When both sensors
fail, hybrid keeps 65 % where the pose graph falls to 44 %.

**In the app.** `scripts/visloc_traj.py --filter hybrid` (also `pf`, `kf_reanchor`,
`kf_window`, `pgo`) runs real visual odometry on the trajectory frames (`measure_track`) and
fuses it with the full per-tile similarity map. The Benchmark Console's Trajectory tab offers
these filters next to the IMU EKF; Studio stays on the EKF, because a drawn route has no images.

End to end on r10 with DenseUAV-ViT (AVL alone: median 405 m, 19 % within 100 m):

| Filter | Result |
|---|---|
| IMU EKF | diverges by kilometres (141 of 143 fixes gated) |
| hybrid | **88 % within 100 m, median 57 m, max 165 m** |

Compute: the filter costs ~60 ms per frame on this CPU. VO costs 0.5–1.3 s per frame
(SIFT at 1200 px, CPU); on a Jetson it should use GPU features.

## Recommendation

- Run **VIO continuously** on board, at camera rate, and not only on AVL frames.
- Use the **hybrid filter**: the particle filter as the global layer, for relocalization, AVL
  outages and no known start; a gated Kalman filter as the fallback when the PF is not
  confident; pure VIO/IMU propagation with growing covariance during sustained AVL outages
  (as in r06).
- Get **k from the lens datasheet**. A wrong k shows up directly as a VO scale error, and VO
  over a few hundred metres of known ground is a cheap way to check it.

## Caveats

- The VO is real, but the IMU coast across failed VO steps is simulated, and so are the outage
  blocks (apart from the real r06 tail).
- The PF temperature (β = 0.5, flat between 0.3 and 0.7), the reanchor thresholds and the
  hybrid's output mode were chosen on these same regions, so there is some tuning leakage.
- Only 144 frames per region were used. r11's frames are not contiguous (half the images are
  missing on disk).
- The AGL for VO scale uses the DEM at the GPS position. On a vehicle it would use the
  estimated position, which matters little at these altitudes.
