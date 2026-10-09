---
title: "GPS-Denied UAV Localization — State of the Art and Improvement Roadmap"
subtitle: "Encoders, search, matching, preprocessing, VO/VIO, fusion, datasets and Jetson compute"
date: "2026-10-02"
---

# GPS-Denied UAV Localization — State of the Art and Improvement Roadmap

**Scope.** This document surveys the published state of the art (to October 2026) for each stage
of an absolute visual localization (AVL) system. It then maps each finding onto this repository
and its measured results, and ends with a prioritized, testable roadmap. Every recommendation
is costed for an NVIDIA Jetson Nano-class board.

**How numbers are tagged.**

- **[M]** measured in this repository (`artifacts/visloc`, 144-query UAV-VisLoc sets, top-1 within 100 m unless stated).
- **[L]** reported by the cited paper and not reproduced here.
- **[P]** a projection made by this document. The method is in §9.2. Treat it as a planning number until it is measured on the board.

---

## 0. Summary

**The premise is right, and the field agrees with it.** Every system from 2023–2026 that
localizes a real UAV against satellite imagery pairs a retrieval or matching front end with
odometry. Examples are FoundLoc, NaviLoc, NGPS, AerialVL and OrthoTrack. None of them relies on
per-frame retrieval alone. No single encoder generalizes to every region, and this repository's
own numbers show it: MegaLoc with heading and centering scores 74.3 % on r05 but 36.8 % on r10
[M].

**Most of the benefit of extra sensors comes before the filter.** Sensors can make the query
look like the map and shrink the area that has to be searched. This attacks the domain gap at
its source:

| Sensor | Removes | Evidence |
|---|---|---|
| Heading (IMU/magnetometer) | rotation gap | [M] MegaLoc r05 56.9 → 68.1 %, ensemble 80.6 → 85.4 % |
| Altitude (baro/rangefinder + terrain model) | scale gap | [L] +41.5 pts R@1 from altitude-based scale normalization |
| Attitude (roll/pitch) | perspective gap | [L] AnyVisLoc: pitch error < 5° has no effect; 30° costs 4.9 pts |
| INS/VIO position + covariance | search area (distractors) | [L] NGPS, STHN, FoundLoc all search only near the prior |

**Ten changes, in priority order:**

| # | Change | Kind | Evidence | Jetson cost | Effort |
|---|---|---|---|---|---|
| 1 | Use per-frame **altitude above ground** to normalize query scale, replacing the blind `--query-scales` search | preprocessing | [L] +41.5 R@1 | saves 2–3 encodes | S |
| 2 | Make **prior-window retrieval** the online default, and report recall *inside the window* as the main KPI | search | [L] used by every fused system | ≈ 0 | S |
| 3 | Add a **multi-hypothesis layer** (particle filter or sliding-window trajectory consensus) between retrieval and the ESKF | fusion | [L] NaviLoc 16× better than per-frame; [M] the ESKF lock-out failure | < 1 ms CPU | M |
| 4 | Test **generalizable learned matchers** (MatchAnything, GIM, MINIMA, SuperPoint+LightGlue, XFeat) for reranking and metric refinement | matching | [L] AnyVisLoc: RoMa 70 % vs SIFT 44 % within 5 m; [M] SIFT verifies 0/10 here | 10–100 ms on Orin | S–M |
| 5 | Add **CAMP** and **DINOv3-SAT ViT-L** to the encoder zoo | encoder | [L] CAMP best of 16 retrievers in AnyVisLoc; DINOv3-SAT is the only satellite-pretrained foundation backbone | 50–90 ms on Orin | S |
| 6 | Replace the simulated IMU with **real sequences** (AerialVL, Nardo-Air) | data | — | — | M |
| 7 | Run odometry in the cheapest place that works: **flight-controller EKF2 or homography VO** on a Nano; **OpenVINS or cuVSLAM** on an Orin | VO/VIO | [L] Jetson VIO benchmarks | 0 → 1 CPU core | M |
| 8 | Derive measurement covariance from **match verification** (inlier count), never from the retrieval score | fusion | [L] NGPS 2.94 m RMSE; [M] score does not predict error | ≈ 0 | S |
| 9 | Build an **ONNX → TensorRT FP16** online path and measure on the board | deployment | — | — | M |
| 10 | For the Nano, train a **label-free distilled ViT-S student** of the encoder ensemble | encoder | [L] D²-VPR: −62.6 % FLOPs vs its teacher | ~50 ms on Nano [P] | L |

**On hardware:** with VIO bridging the gaps, AVL only needs about one fix every 1–2 s. That
leaves a whole second of GPU time per fix. The original Jetson Nano can afford one ViT-B-class
encoder per fix when heading is known (~0.5 s [P]). An Orin Nano Super runs the full
three-encoder ensemble in ~0.2 s [P]. The original Nano's real limits are its CPU, which is too
weak for VIO alongside everything else, and its software: JetPack 4.6 with Python 3.6 cannot
install this repository's stack. Choose the **Orin Nano Super 8 GB** if there is any choice.

---

## 1. Where the project stands

**What is built.**

- **Encoders.** About 15, including DenseUAV-ViT, MixVPR, CosPlace, EigenPlaces, AnyLoc (lite/L/G/sat), DINOv2/v3, Sample4Geo, Game4Loc, MegaLoc, BoQ and InfoGeo.
- **Search.** FAISS Flat, HNSW and IVF-PQ.
- **Geo-fusion.** Five top-5 → one-pose methods.
- **Training-free lifts.** Domain centering and heading-based rotation selection.
- **Reranking.** SIFT/ORB/AKAZE with a homography plausibility guard.
- **Navigation.** A 15-state loosely coupled ESKF fed by a *simulated* IMU, plus GUI Trajectory and Studio tabs.

**Best measured retrieval** (top-1 within 100 m, 144 queries per region) [M]:

| Configuration | GMAC/query | r05 | r10 | r06 |
|---|---:|---:|---:|---:|
| MegaLoc, 4 rotations, raw | 184 | 56.9 % | 16.0 % | — |
| MegaLoc + heading + flight centering | 46 | **74.3 %** | 36.8 % | 43.1 % |
| InfoGeo-dense + heading | ~92 | 72.2 % | 25.0 % | 27.8 % |
| MegaLoc + Game4Loc + AnyLoc-L + heading | 173 | **85.4 %** (geo-fused 91.7 %) | — | — |
| Best pair seen on r10 (MegaLoc + AnyLoc-L) | 124 | — | 47.2 % | — |

**The gap.** The best configuration scores 74–85 % in one region and 37–49 % in the others.
That gap is the generalization problem, measured. It is not a bug to tune away.

---

## 2. Why encoders fail across regions, and which sensor fixes which part

The cross-region drop is not one problem. It is at least seven, and most of them are geometric,
which onboard sensors can remove.

| Gap source | What happens | Remedy | Evidence |
|---|---|---|---|
| **Rotation** | Drone yaw ≠ north-up tiles | Rotate the query by IMU/compass heading | [M] +11 pts on r05; [L] AnyVisLoc: 60° yaw noise costs 25.7 pts |
| **Scale** | Altitude and terrain change the footprint; tiles have a fixed size | AGL = baro altitude − DEM, then crop to the tile footprint | [L] +41.5 pts R@1 |
| **Perspective** | Camera not nadir | Warp to nadir using roll/pitch | [L] AnyVisLoc: < 5° negligible, 30° → −4.9 pts |
| **Appearance / time** | Season, satellite capture date, sun angle | Foundation features, domain centering, recent maps, thermal at night | [M] centering: MegaLoc +16 pts on r05 |
| **Map quality** | Satellite (0.2–0.6 m/px, old) vs aerial ortho (0.07 m/px + DSM) | Use the best map that exists for the area | [L] AnyVisLoc: 74.1 % vs 18.5 % within 5 m |
| **Self-similarity** | Farmland, forest, desert and water look alike everywhere | Cannot be fixed per frame; needs trajectory consensus and a search window | [L] NaviLoc 16× vs per-frame state of the art; TRAIL +8.3 pts |
| **Search area** | 380 km² means thousands of distractor tiles | INS/VIO prior window | [L] STHN: 512 m radius; NGPS: VIO-predicted region |

**Per-frame top-1 is the wrong KPI for a fused system.** After fusion, three properties matter:

1. **Recall inside the prior window.** Is the true tile in the top-K among tiles within radius *r* of the INS estimate? Retrieval among 50 nearby tiles is a much easier problem than among 5,000.
2. **Temporal independence of errors.** If wrong answers scatter while correct answers cluster along the motion, consensus over a few fixes recovers the truth even when per-frame top-1 is 40 %.
3. **Detectable failure.** A confidence signal that separates good fixes from bad ones. The retrieval cosine score is *not* one here [M]. Geometric verification and motion consistency are.

**Recommendation:** add "R@1/R@5 within *r* = 250/500/1000 m of the prior" and "precision of
accepted fixes" to `visloc_eval.py`'s summary. Use them to choose encoders.

---

## 3. Encoders (global descriptors)

### 3.1 The landscape

GMAC = multiply-accumulates per image at the stated input. Values marked [M] are measured in
this repository; the others are computed with the formula in §9.2 or reported by the paper.

| Model | Year / venue | Backbone @ input | GMAC | Trained on | In repo | Notes |
|---|---|---|---:|---|---|---|
| **Generic VPR (street-level training)** | | | | | | |
| CosPlace | CVPR 2022 | ResNet @322 | 17.5 [M] | SF-XL | yes | |
| EigenPlaces | ICCV 2023 | ResNet-50 @322 | 9.1 [M] | SF-XL | yes | |
| MixVPR | WACV 2023 | ResNet-50 @320 | 8.1 [M] | GSV-Cities | yes | |
| SALAD | CVPR 2024 | DINOv2-B/14 | ~50 | GSV-Cities | via MegaLoc | optimal-transport aggregation |
| BoQ | CVPR 2024 | DINOv2-B / RN-50 | ~50 / ~8 | GSV-Cities | yes (weights incomplete) | learnable query aggregation |
| CliqueMining | ECCV 2024 | DINOv2-B + SALAD | ~50 | GSV-Cities + mined | no | better hard-negative mining |
| **MegaLoc** | CVPRW 2025 | DINOv2-B/14 @322 + SALAD | 45.9 [M] | 5 datasets combined | yes | **best single generalist here [M]** |
| D²-VPR | AAAI 2026 | distilled VFM + deformable aggregator | −62.6 % vs CricaVPR [L] | — | no | candidate for the Nano |
| **Training-free, foundation-based** | | | | | | |
| AnyLoc | RA-L 2023 | DINOv2 + VLAD | 21.9 (lite) / 77.8 (L) [M] | none (vocabulary only) | yes | FoundLoc reports 94 % R@1 on aerial data [L] |
| DINOv2 / DINOv3 (web) + GeM | 2023 / 2025 | ViT-S…L | 4.7–81 | LVD-142M / 1689M | yes (dinov3-b/-l) | |
| **DINOv3-SAT ViT-L/16** | 2025 | ViT-L/16 @224 | ~63 [P] | **493 M Maxar satellite images** | **no** | the only satellite-pretrained SSL backbone released |
| VFM-Loc | arXiv 2603.13855 | VFM + GeM + SW-RMAC | backbone | none | no | caution: its Procrustes step fits on test pairs (label leak) [M] |
| **UAV ↔ satellite, supervised** | | | | | | |
| DenseUAV-ViT | TIP 2024 | ViT-S/16 @224 | 4.2 [M] | DenseUAV | yes | excellent in-domain, weak elsewhere [M] |
| Sample4Geo | ICCV 2023 | ConvNeXt-B @384 | ~45 | University-1652 | yes | |
| MCCG / DAC / QDFL | 2023–2024 | ConvNeXt | ~45 | University-1652 | no | AnyVisLoc R@1: DAC 58.3, QDFL 56.5 [L] |
| **CAMP** | TGRS 2024 | ConvNeXt | ~45 | University-1652 / SUES-200 | **no** | **AnyVisLoc R@1 62.4 %, best of 16 [L]**; +8.85 pts cross-dataset [L] |
| Game4Loc | AAAI 2025 | ViT-B | 49.3 [M] | GTA-UAV (synthetic) | yes | synthetic training that transfers to real imagery |
| EGS | arXiv 2509.20684 | rotation-equivariant | — | University-1652 | no | built for cross-dataset generalization |
| InfoGeo | ICML 2026 | DINOv2 + MixVPR head | 91.7 [M] | University-1652 / GTA | yes | its r05 lead did **not** carry over to r10/r06 [M] |
| CAEVL | WACV 2026 | lightweight | 1.4 [L] | CAEVL | no | **no public weights** |
| Bearing-UAV | CVPR 2026 | global + local structure | — | Bearing-UAV-90K | no | predicts **location and heading**; code promised |
| **Adaptation using the map only (no query labels)** | | | | | | |
| Reference-Set Fine-tuning | arXiv 2510.03751 | any | — | the map's own tiles | — | +2.3 pts R@1 on average [L] |

### 3.2 What the evidence says about generalization

- **No model wins every region.** That holds both in this repository [M] and in the AnyVisLoc benchmark. AnyVisLoc found that "training/sampling strategies proved more effective than architecture changes alone" [L].
- **Papers' generalization claims do not transfer reliably.** InfoGeo's cross-view generalization did not carry over to r10/r06 [M]. Evaluate every new model on all three regions before adopting it.
- **Ensembles of models with different training data are the most consistent lift** [M]. Their errors are complementary, but their compute adds up (§9).
- **Foundation features work best on a normalized query.** FoundLoc's 94 % R@1 came with flights that held the yaw north-aligned [L]. That matches the heading result here [M].
- **Release details matter.** DINOv3's satellite weights exist only as ViT-L/16 (300 M parameters) and ViT-7B ([official repo](https://github.com/facebookresearch/dinov3)). Some 2026 blog posts claim a satellite ConvNeXt-T; the official release has none.

### 3.3 Encoder recommendations

1. **Keep the base.** MegaLoc + heading + flight centering is the most consistent single model across r05/r10/r06 [M].
2. **Evaluate next, in this order:**
   - **CAMP.** A ConvNeXt CNN, which is TensorRT-friendly. Code is [public](https://github.com/Mabel0403/CAMP); confirm that the checkpoint downloads.
   - **DINOv3-SAT ViT-L/16** with GeM, and with AnyLoc-style VLAD using a vocabulary built from map tiles. This stays training-free.
   - **CliqueMining.**
   
   Run each with heading + flight centering on r05/r06/r10 and report the in-window KPIs from §2.
3. **For the Nano, distill the ensemble into one small student** (ViT-S/16 or ConvNeXt-T, ~5 GMAC, ~50 ms on the Nano [P]).
   - Train with a feature-regression loss on *unlabeled* imagery: satellite tiles from many regions plus synthetic drone-like augmentation (rotation, scale, blur, colour, haze).
   - This uses no GPS labels and no target-region queries, so it does not specialize to a dataset the way the stopped fine-tune did. D²-VPR [L] shows that this kind of distillation keeps most of the teacher's accuracy.
4. **Do not use:** PCA-whitening fitted on the map (hurts [M]), or per-dataset supervised heads.

---

## 4. Search

### 4.1 Index structure

| Index | When | Memory for N tiles × D dims | Notes |
|---|---|---|---|
| **Flat (exact)** | N ≤ ~100 k | N·D·2 bytes in FP16 | Ardabil 3,054 tiles × 8,448-D = 52 MB; UAV-VisLoc scale is fine |
| HNSW | N ≥ 100 k, latency-critical | +~10 % | approximate |
| IVF-PQ | N in the millions, memory-critical | ~N·64 bytes | lossy; costs recall |
| **Prior-window masked Flat** | **always, once a prior exists** | same | only tiles within radius *r*: hundreds of dot products, effectively free |

For the city- and region-scale maps this project targets, exact search is cheap. The memory
limit on a 4 GB Nano comes from descriptor dimension, not tile count: 50 k tiles × 8,448-D in
FP32 is 1.7 GB. Store descriptors in FP16. If needed, test plain PCA *without* whitening to
1,024-D.

### 4.2 Prior-constrained search

This is the most important search change.

- **Window radius.** Use *r* = max(*r*_min, *k*·σ_pos), where σ_pos comes from the filter covariance. NGPS predicts the search region from VIO velocity [L]. STHN searches a 512 m radius around the last known position [L].
- **Already partly built.** `visloc_eval.py --prior-sigma` exists; it should become the online default in `avl/retrieval.localize`.
- **Two states are required:**
  - *Tracking*: search the window.
  - *Lost / relocalize*: search globally and hand over to the multi-hypothesis layer (§7) when the filter has rejected fixes for *T* seconds or σ_pos exceeds a limit. The window must widen while the INS coasts.

### 4.3 Coarse-to-fine localization

The standard 2025–2026 pipeline (AnyVisLoc, the hierarchical AVL system of *Remote Sensing*
2025, NGPS, OrthoLoC) has three stages:

1. Global retrieval finds candidate tiles.
2. Local matching finds pixel correspondences.
3. A pose solver computes the position: a homography for flat ground, or PnP when a DSM is available.

Retrieval gives about one tile of accuracy (tens of metres). Matching gives metres, and an inlier
count that is a far better confidence signal than the cosine score.

### 4.4 Reranking and matching

Numbers are from AnyVisLoc, using CAMP retrieval and an **aerial photogrammetry map**; runtimes are
on a desktop GPU [L]:

| Matcher | Within 5 m | Within 10 m | Within 20 m | Runtime | Jetson fit |
|---|---:|---:|---:|---:|---|
| RoMa (dense) | 70.1 % | 81.3 % | 87.6 % | 659 ms | Orin only, keyframes only |
| DKM | 65.6 % | 80.2 % | 86.4 % | 4,915 ms | no |
| LoFTR + GIM weights | 59.5 % | 74.1 % | 81.6 % | 165 ms | Orin |
| SuperPoint + LightGlue + GIM | 57.0 % | 75.4 % | 84.8 % | 105 ms | **Orin: LightGlue TensorRT FP16 7.9 ms at 512 keypoints [L]** |
| SIFT | 43.9 % | 55.7 % | 62.5 % | 316 ms | CPU; **0/10 verified on this repo's satellite data [M]** |
| ORB | 3.9 % | 8.0 % | 13.6 % | 44 ms | useless cross-domain |

**On a satellite map**, the same AnyVisLoc pipeline drops to 18.5 / 38.7 / 58.5 % (within 5 / 10 /
20 m) [L]. So with satellite maps, plan for ~20 m accuracy, not 5 m.

**Matchers built for cross-domain generalization**, none tested here yet:

- **MatchAnything** (2025). Cross-modality pre-training on RoMa and ELoFTR bases. Large gains on visible↔thermal and visible↔SAR aerial pairs [L].
- **MINIMA** (CVPR 2025). Same idea, using a generative data engine.
- **GIM** (ICLR 2024). Self-training on internet video. Its weights are the "GIM" rows in the table above.
- **XFeat + LighterGlue** (CVPR 2024). The lightest option: 27 FPS on a laptop CPU at VGA resolution [L]. The only one plausible on the original Nano. Its cross-domain accuracy is unknown.

**Recommendation:** run all of these through `avl/rerank.py` on r05/r10. Keep
`_is_plausible_homography` (see the memory note on degenerate homographies). Install with
`pip install -e '.[rerank]'` once network access is available. The question to answer is "how
many of the top-10 are verified, and what is the post-homography error". If a matcher verifies
even 30–50 % of queries, the inlier count becomes the confidence that §7 needs.

### 4.5 Sequence and trajectory-level search

This is the highest leverage for the least compute.

- **Classic sequence matching** (SeqSLAM, SeqNet) assumes an *ordered* reference sequence. A tile grid is not ordered, so these do not apply directly.
- **TRAIL** (ECCV 2026). A CRF over candidate references that combines visual similarity with camera-motion consistency, against an **unordered** database. Up to +8.3 pts, transfers without retraining, and helps most where visual cues are scarce [L].
- **NaviLoc** (*Drones* 2026). Training-free. It treats VPR as a noisy measurement and VIO as a relative-motion prior:
  - Stage 1: a global SE(2) alignment of the VIO trajectory to the VPR candidates.
  - Stage 2: sliding-window weighted Procrustes refinement.
  - Result: 19.5 m mean error on 50–150 m rural flights, 16× better than the per-frame state of the art, running at 9 FPS on a Raspberry Pi 5 [L].
- **FoundLoc** (2023). ICP alignment of a sliding window of keyframes to the retrieved positions, with DBSCAN keeping the largest cluster, all inside an EKF [L].

### 4.6 Move rotation and scale search to the map side

When heading or altitude is unknown (at take-off, or with a disturbed magnetometer), store
descriptors of rotated and scaled tile variants offline. Four rotations × three scales makes the
index 12× larger but keeps the online cost at **one** encode. Rotating the tile is not exactly
equivalent to rotating the query for a non-invariant encoder, so measure it before relying on it.
Once heading is known, use the single heading-matched rotation, which beats the four-rotation
search [M].

---

## 5. Preprocessing

### 5.1 Query side, in pipeline order

1. **Image-quality gate.** Skip the fix when the frame is blurred, low-texture (water, uniform fields, cloud) or saturated. This saves compute, and it removes the garbage fixes that destabilize the filter. A Laplacian-variance and gradient-entropy check costs about 1 ms.
2. **Undistort** with calibrated intrinsics.
3. **Nadir rectification.** If the camera is not on a stabilized nadir gimbal, warp with the rotation-only homography *H = K·R·K⁻¹*, built from the IMU roll/pitch.
4. **North alignment** by heading. Already built [M]. Note that the GUI's "1 (heading known)" setting does not pass `--north-align`.
5. **Scale normalization.** This is new and has the largest expected gain.
   - AGL = barometric/GNSS-free altitude − DEM(prior lat, lon), using SRTM or Copernicus GLO-30 at 30 m.
   - Ground footprint = 2 · AGL · tan(FOV/2). Crop or resize the query so its footprint matches the tile's ground size.
   - UAV-VisLoc's `height` column is **above sea level** (≈ 2,315 m on r05), so a DEM is required.
   - Today the pipeline matches tile size to footprint *per region* (`visloc_prepare.py`) and searches query crops blindly (`--query-scales`). One informed scale should be more accurate and also two to three times cheaper. The altitude-adaptive paper reports +41.5 pts R@1 [L].
6. **Photometric normalization.**
   - Domain centering is built [M]; "flight" mode is the most consistent.
   - Optionally add histogram matching to the map's statistics, and CLAHE for haze.
   - PCA-whitening hurts [M].

### 5.2 Map side

These choices often matter more than the encoder.

- **Map source.** Use the best orthophoto available: aerial ortho + DSM beats satellite by 4× within 5 m [L]. Governmental orthophotos and DSMs, as in OrthoLoC, are the gold standard where they exist.
- **Capture date and season.** Use recent imagery from the same provider. Where seasons differ strongly, index several epochs (summer and winter) as separate entries.
- **Tiling.** Cut tiles to match the *expected* altitude footprint, with ~50 % overlap. Add one or two coarser levels if altitude varies a lot during the mission.
- **Elevation data.** Ship a DEM/DSM with the map. It is needed for AGL (scale) and for PnP.
- **Informativeness prior.** Down-weight tiles that are water or uniform cover, because they produce confident wrong matches.
- **Storage.** Precompute everything offline and store descriptors in FP16.

### 5.3 Night and thermal

- **STHN / UASTHN** (ICRA 2025): thermal-to-satellite deep homography with uncertainty from crop test-time augmentation and ensembles. 7 m error within a 512 m search radius, with a 97 % success rate for the uncertainty estimate [L].
- **MatchAnything and MINIMA** cover thermal↔visible matching generally.
- Relevant datasets are Boson-nighttime and IRVL328 (§8).

---

## 6. Visual and visual-inertial odometry

**Physics first.** The right odometry depends on altitude:

- **Low altitude (≤ ~150 m).** Monocular VIO works: there is enough parallax, and a rangefinder reaches the ground.
- **High altitude (400–2,000 m, like UAV-VisLoc).** There is almost no parallax. Monocular scale is unobservable from vision, stereo baselines are useless, and rangefinders do not reach the ground. The ground is effectively a plane, so **homography VO scaled by altitude (barometer + DEM)** is the correct model.

A 2026 evaluation of monocular SLAM on high-altitude nadir footage found that no system kept the
global trajectory shape on long sequences, and vertical estimates were poor. The best results
were MASt3R-SLAM at 0.53 % of path length on short flights and DROID-SLAM at 2.88 % on average
[L]. VO/VIO bridges the gaps between absolute fixes; it cannot replace them.

| Method | Type | Sensors | Jetson evidence | Fit here |
|---|---|---|---|---|
| **PX4 EKF2 + optical flow + rangefinder** | filter on the flight controller | flow sensor, rangefinder, IMU, barometer | **zero Jetson cost** | best low-altitude choice for the original Nano; range-limited |
| **Homography VO** (KLT on GPU via NVIDIA VPI or OpenCV) + barometer/DEM | planar VO | nadir camera, barometer, DEM | very light | **best high-altitude choice; feeds the existing ESKF** |
| RaD-VIO | rangefinder-aided downward VIO | camera, IMU, rangefinder | light | low-altitude nadir |
| ROVIO | photometric EKF | mono + IMU | lowest CPU of seven methods on TX2/Xavier [L] | Nano-plausible |
| OpenVINS | MSCKF | mono/stereo + IMU | efficient; no published Nano figures | Orin; Nano borderline |
| VINS-Mono / Fusion | optimization | mono/stereo + IMU | ~150–170 % CPU on AGX Xavier [L] | Orin; too heavy beside AVL on a Nano |
| ORB-SLAM3 | optimization + map | mono/stereo + IMU | Jetson GPU port exists (2026) | Orin |
| Basalt / Kimera | optimization | stereo + IMU | — | stereo is not useful at altitude |
| AirSLAM | learned points + lines, TensorRT | mono/stereo (+ IMU) | 40 Hz on an embedded board [L] | Orin; robust to lighting changes |
| **cuVSLAM** (NVIDIA Isaac ROS) | CUDA VO/SLAM | 1–32 cameras + IMU | 1.8 ms tracking on AGX Orin, stereo 768×480 [L] | **Orin (JetPack 6) only** |
| LEVIO | ORB + bundle adjustment | mono + IMU | 20 FPS at < 100 mW on a RISC-V SoC [L] | shows how light VIO can get |
| DPVO, DROID-SLAM, MASt3R-SLAM | learned | mono | ≥ 3–4 GB GPU memory | no |

**Recommendations:**

- **Original Nano.** Leave odometry on the flight controller (PX4 EKF2 with IMU, barometer, and flow + range at low altitude). Add a GPU homography VO for high altitude. The Jetson then runs only AVL and fusion.
- **Orin Nano Super.** OpenVINS or cuVSLAM mono-inertial, alongside AVL.
- **Both.** Keep the existing ESKF as the integrator and feed VO into it as a velocity or Δpose measurement. That is the "filters.py interface left open" in the design document.

---

## 7. Fusion

### 7.1 Options

| Method | Multimodal? | Starts with unknown position? | Compute | Used by |
|---|---|---|---|---|
| ESKF, loosely coupled (**this repo**) | no | no | trivial | many |
| UKF with adaptive covariance from match quality | no | no | trivial | **NGPS**: 2.94 m RMSE at 60–150 m, 3.5× better than VIO alone, Orin NX [L] |
| **Particle filter over the map** | **yes** | **yes** | 1–10 k particles, well under 1 ms | Jurevičius et al. (2.6× more accurate than ORB-SLAM2 in simulation), Kinnari et al. (season-invariant orthophoto matching) [L] |
| **Sliding-window trajectory alignment** (Procrustes/ICP + RANSAC or DBSCAN) | **yes (by consensus)** | **yes** | cheap | **NaviLoc** (Raspberry Pi 5), **FoundLoc** [L] |
| CRF / HMM over tiles | yes | yes | moderate | TRAIL [L] |
| Factor graph (GTSAM iSAM2) + robust kernels / switchable constraints / GNC | partly (robust costs) | needs initialization | moderate CPU | long-term option |
| Tightly coupled (map correspondences as residuals) | no | no | needs a working matcher | OrthoTrack (ECCV 2026), PiLoT v2 [L] |

### 7.2 Diagnosis of the current ESKF

The memory notes for this project record the failure: clamping *P* or tightening the χ² gate
makes good fixes get gated after a coast, and the filter diverges. Re-initializing on a run of
gated fixes re-centres the filter on garbage [M]. This is the textbook failure of a **unimodal
Gaussian filter fed by a sensor that is right 40–75 % of the time with non-Gaussian
(teleporting) errors**. Tuning cannot fix it. The cure is to postpone the decision until the
evidence is consistent over time.

### 7.3 Proposed two-layer architecture

```
 IMU 100-200 Hz ──► INS / ESKF (unchanged, 15-state) ──► pose @ IMU rate
                         ▲  validated fix (μ, Σ)
                         │
 VO / VIO 10-20 Hz ──────┤ (velocity / Δpose updates)
                         │
 Camera ─► preprocess ─► encoder ─► window search (top-K) ─► [ multi-hypothesis layer ]
                                                             PF or sliding-window consensus
                                                             emits a fix only when unimodal
```

- **Layer A: multi-hypothesis, about 1 Hz.**
  - Particle state: (x, y) plus an optional heading bias.
  - Prediction: the VO/INS Δp, with noise proportional to distance travelled. This also handles the 36 s frame gaps in r05.
  - Likelihood of particle *p*: Σₖ wₖ · N(p; cₖ, σₜ²) + ε, where cₖ are the top-K tile centres, wₖ = softmax(sₖ/τ), and the floor ε models "all candidates wrong".
  - Resample when the effective sample size drops below N/2.
  - Emit (μ, Σ) to the ESKF only when the particle cloud is unimodal and its spread is small. Otherwise emit nothing.
  - With 2,000 particles and K = 20, that is about 40 k Gaussian evaluations per fix: microseconds on any CPU.
  - Fits the existing `avl/nav/filters.py::FILTERS` registry.
- **Alternative to Layer A.** NaviLoc-style sliding-window SE(2) alignment of the last N VO poses to their candidate sets, with RANSAC over candidates. Deterministic and easy to debug.
- **Layer B.** The existing ESKF, unchanged. Its measurement noise R comes from Layer A's spread, or from the matcher's inlier statistics once §4.4 works. Never take R from the retrieval cosine score [M].
- **Relocalization.** If Layer A has no consensus for *T* seconds, widen the search window to global and re-seed the particles from global retrieval.

### 7.4 Longer term

Move to a sliding-window factor graph: GTSAM iSAM2 with IMU pre-integration, VO factors, and VPR
or matching prior factors with switchable constraints or graduated non-convexity (GNC).
"Holistic Fusion" (2025) is a good reference for a setup-agnostic version. The dense alternative,
OrthoTrack, matches keyframes to an orthophoto, lifts them to 3D with the DSM, and propagates by
optical flow, running in real time on one GPU [L]. It becomes worthwhile only after a matcher
verifies reliably on your maps.

---

## 8. Datasets

| Dataset | Year | Altitude | Size / coverage | Reference map | IMU / sequences | Use it for |
|---|---|---|---|---|---|---|
| **Retrieval (single frames)** | | | | | | |
| University-1652 | 2020 | synthetic orbit | 1,652 buildings | satellite | no | standard training and evaluation |
| SUES-200 | 2023 | 150–300 m | 200 sites, 24 k images | satellite | no | altitude generalization |
| DenseUAV | 2024 | 80–100 m | 14 campuses | satellite | no | in repo |
| **UAV-VisLoc** | 2024 | 400–2,000 m | 11 regions, 380 km² | satellite mosaic | time-ordered, attitude, no raw IMU | in repo; leave-region-out evaluation |
| GTA-UAV (Game4Loc) | 2025 | 80–650 m | 33.8 k drone images, 81.3 km² | satellite, 4 zoom levels | synthetic | partial-overlap training |
| **AnyVisLoc** | 2025 | ~6–500 m | 24 scenes, ~20 k images | aerial ortho + satellite + 2.5D | no | **the best end-to-end retrieval → matching → PnP benchmark** |
| Bearing-UAV-90K | 2026 | — | multi-city | satellite | — | location + heading |
| LASED / CAEVL / WildNav | 2022–2025 | various | Estonia ~1 M images / non-nadir / wilderness | ortho / satellite | no | out-of-domain checks |
| **Sequences with IMU and a satellite map** (to test fusion) | | | | | | |
| **AerialVL** | RA-L 2024 | various | 11 sequences, ~70 km | satellite | **yes** | **VPR + alignment + VO baselines; first choice for real-IMU fusion** |
| **Nardo-Air** (FoundLoc) | 2023 | 50–100 m | 20+ trajectories | satellite, 0.1 m/px | **yes** | real-world VIO + VPR |
| SatLoc | 2025 | 100–300 m | rotorcraft | satellite | multi-sensor, synchronized | hierarchical fusion |
| VPAIR / ALTO | 2022 | high / helicopter | 107 km / ~410 km | ortho + DSM / LiDAR | trajectory | long high-altitude traverses |
| **6-DoF localization with orthophoto + DSM** | | | | | | |
| **OrthoLoC** | NeurIPS 2025 | low | 16,425 images, 47 locations, 19 cities | governmental ortho + DSM | listed as yes | metric 6-DoF; AdHoP refinement |
| CrossLoc / LoD-Loc / UAVD4L | 2022–2024 | low | — | ortho+DSM / LoD city models | — | 3D-map variants |
| **Odometry only** | | | | | | |
| EuRoC, UZH-FPV, KAIST VIO, TartanAir, Mid-Air | 2016–2021 | indoor / synthetic | — | — | yes | VIO sanity checks |
| MARS-LVIG, UAVScenes, NTU-VIRAL | 2022–2025 | outdoor, some downward | — | RTK GNSS | yes | downward-looking VIO |
| **Thermal / night** | | | | | | |
| Boson-nighttime (STHN) | 2023 | high | 33 km² thermal, 216 km² satellite | satellite | — | night operation |
| IRVL328 / SkyPin | 2025–2026 | 100–200 m | — | 2.5D | — | infrared ↔ visible |

**Evaluation protocol for "does it generalize":** report **leave-one-region-out** results. Use all
11 UAV-VisLoc regions plus AnyVisLoc scenes, and never pick hyper-parameters on the region being
scored. Report median, p95 and the in-window KPIs (§2), not just top-1 on r05.

**For the real target** (Ardabil, see the coverage report): fly a few sorties that log
time-synchronized camera, IMU, barometer and magnetometer data against RTK ground truth. A
30-minute dataset from the real area, real camera and real altitude is worth more than any
public benchmark for the final decision.

---

## 9. Compute on the Jetson

### 9.1 Boards

| | Jetson Nano (original) | Jetson Orin Nano Super 8 GB |
|---|---|---|
| GPU | 128-core Maxwell, no tensor cores | 1,024-core Ampere + 32 tensor cores |
| Peak | 472 GFLOPS FP16 | 67 TOPS sparse / 33 TOPS dense INT8; NVIDIA lists 17 FP16 TFLOPS |
| CPU | 4 × Cortex-A57 @ 1.43 GHz | 6 × Cortex-A78AE @ 1.7 GHz |
| Memory | 4 GB LPDDR4, 25.6 GB/s, shared | 8 GB LPDDR5, 102 GB/s, shared |
| Software | **JetPack 4.6 max: Ubuntu 18.04, Python 3.6, CUDA 10.2, TensorRT 8.2** | JetPack 6: Ubuntu 22.04, Python 3.10, TensorRT 10 |
| Repo stack (torch ≥ 2.1, timm ≥ 1.0, Python ≥ 3.10) | **cannot install**; the online path must be rewritten as TensorRT engines | runs as is |
| Power | 5–10 W | 7 / 15 / 25 W |

### 9.2 Projection method

- **MACs.** For a ViT, MACs ≈ L · (12·N·D² + 2·N²·D) + patch embedding. This formula reproduces the measured values within ~10 % (MegaLoc 50.4 vs 45.9 measured; AnyLoc-L 81.0 vs 77.8).
- **Jetson Nano anchor.** NVIDIA measured ResNet-50 (4.1 GMAC) at 36 FPS with TensorRT FP16, which is 148 GMAC/s achieved. Maxwell has no tensor cores and handles attention poorly, so ViTs are assumed to reach **60 %** of that.
- **Orin Nano Super anchor.** ViT-B/16 TensorRT FP16 at 19.9 ms (17.6 GMAC), taken from §8.4 of the coverage report, plus 1.2 ms fixed overhead.

**Projected encode latency, batch 1, one rotation [P]:**

| Encoder | GMAC | Jetson Nano | Orin Nano Super |
|---|---:|---:|---:|
| DenseUAV-ViT (ViT-S/16 @224) | 4.2 | ~50 ms | ~6 ms |
| DINOv3 ViT-S/16 @224 (distillation student) | 4.7 | ~55 ms | ~7 ms |
| ConvNeXt-T @224 | 4.5 | ~50 ms | ~6 ms |
| DINOv2 ViT-S/14 @224 (AnyLoc-lite class) | 6.1 | ~70 ms | ~8 ms |
| **MegaLoc** (ViT-B/14 @322) | 45.9 | **~520 ms** | **~53 ms** |
| CAMP / Sample4Geo (ConvNeXt-B @384) | ~45 | ~510 ms | ~52 ms |
| DINOv3-SAT ViT-L/16 @224 | 62.8 | ~710 ms | ~72 ms |
| AnyLoc-L (ViT-L/14 @224) | 77.8 | ~880 ms | ~89 ms |
| **Ensemble** MegaLoc + Game4Loc + AnyLoc-L | 173 | **~2.0 s** | **~0.2 s** |

The coverage report's earlier projection for DenseUAV-ViT on the Nano was 212 ms, scaled down
from the Orin with exponent 1.0. This table, anchored on NVIDIA's measured Nano ResNet-50, gives
~50 ms. The truth is probably in between. **Measure it on the board before deciding.**

### 9.3 Budget for a fused system

With VIO or the flight-controller EKF bridging the gaps, the AVL needs about one fix every
1–2 s. At 15 m/s the vehicle covers 15–30 m between fixes, and VO drifts well under 1 m in that
distance.

| Stage | Jetson Nano | Orin Nano Super |
|---|---|---|
| Decode, undistort, rectify, crop (GPU) | ~15 ms | ~5 ms |
| Encoder (heading known: 1 rotation, 1 scale) | MegaLoc ~520 ms, or a distilled student ~55 ms | ensemble ~200 ms |
| Window search + geo-fusion + Layer A filter | < 2 ms (CPU) | < 1 ms |
| Matcher on top-3 | XFeat-class only, or none | SuperPoint + LightGlue ~3 × 10 ms |
| VO / VIO | on the flight controller (EKF2) or GPU homography VO | OpenVINS / cuVSLAM concurrently |
| **Fix rate** | **~1 fix / 0.6 s** with MegaLoc; ~10 Hz with a student | **~3–4 Hz** with ensemble + matcher |

**Conclusion.** Sensor fusion relaxes the encoder's latency budget by about 10×. That is what
makes a ViT-B-class encoder affordable even on the original Nano. The Nano's binding constraints
are its CPU (VIO and fusion next to Python) and its software stack, not GPU FLOPs.

---

## 10. Roadmap

Each step has a measurement and a decision rule. Phase 0 needs only the data already in the
repository, and it stays within the encoder and retrieval focus.

| Phase | Step | What to run | Decision rule |
|---|---|---|---|
| **0: offline, existing data** | E1. Scale normalization | Download SRTM/GLO-30 for regions 05/06/10; compute per-frame AGL; one informed crop vs `--query-scales` | adopt if r10 and r06 improve, not just r05 |
| | E2. In-window KPIs | `visloc_eval.py --prior-sigma` 250/500/1000 m; add R@5-in-window and accepted-fix precision | becomes the KPI for every later choice |
| | E3. Learned matchers | `pip install -e '.[rerank]'`; SuperPoint+LightGlue, XFeat+LighterGlue, GIM, MatchAnything, RoMa on r05/r10 top-10 | keep any matcher that verifies ≥ 30 % with errors < 30 m |
| | E4. New encoders | CAMP, DINOv3-SAT-L (GeM and VLAD), CliqueMining, each with heading + flight centering | adopt only if it beats MegaLoc on ≥ 2 of 3 regions |
| **1: multi-hypothesis fusion** | E5. Layer A | Particle filter (or NaviLoc-style window alignment) in `avl/nav/filters.py`; Trajectory tab on r05/r10 with real timestamps | fused median and p95 better than VPR-only on r10, where retrieval is weak |
| **2: real IMU** | E6. Real sequences | AerialVL / Nardo-Air: OpenVINS offline → Layer A → ESKF | drift between fixes and fix-acceptance precision on real data |
| **3: Jetson** | E7. TensorRT | ONNX → TensorRT FP16 for the chosen encoder(s) and matcher; time them on the board; distill (§3.3) only if needed | fix period ≤ 1 s with VO running |
| **4: field** | E8. Own flights | Time-synchronized camera, IMU, barometer and magnetometer logs with RTK ground truth over the target area | the final go/no-go numbers |

---

## 11. What not to do

| Don't | Why |
|---|---|
| Search for one encoder that works everywhere | Contradicted by measurements here [M] and by AnyVisLoc [L]; use sensors and fusion instead |
| Fine-tune per dataset | Rejected for generality; use the label-free distillation (§3.3) or map-only adaptation |
| Treat the retrieval cosine score as confidence | It does not predict error here [M] |
| Tighten the χ² gate or clamp P to fight bad fixes | Measured to make divergence worse [M]; add Layer A instead |
| Search four rotations when heading is known | One heading-matched rotation is better and 4× cheaper [M] |
| Fit PCA-whitening on the map | Hurts [M] |
| Run DROID-SLAM, MASt3R-SLAM, VGGT, RoMa or DKM on the original Nano | Not enough memory or compute |
| Expect 5 m accuracy from satellite maps | AnyVisLoc reaches 58.5 % within 20 m on satellite maps [L]; 5 m needs aerial ortho + DSM |

---

## 12. Sources

**Systems, benchmarks and surveys**

- FoundLoc (2023): <https://arxiv.org/abs/2310.16299>
- AnyVisLoc benchmark (2025): <https://arxiv.org/abs/2503.10692>, <https://github.com/UAV-AVL/Benchmark>
- NaviLoc (*Drones* 2026): <https://www.mdpi.com/2504-446X/10/2/97>
- NGPS (2026): <https://arxiv.org/abs/2607.18936>
- AerialVL (RA-L 2024): <https://ieeexplore.ieee.org/document/10632587/>, <https://github.com/hmf21/AerialVL>
- OrthoTrack (ECCV 2026): <https://arxiv.org/abs/2606.25245>
- OrthoLoC (NeurIPS 2025): <https://deepscenario.github.io/OrthoLoC/>
- PiLoT v2 (2026): <https://arxiv.org/abs/2606.31098>
- DECO (2026): <https://arxiv.org/abs/2608.22289>
- Hierarchical AVL for low-altitude drones (*Remote Sensing* 2025): <https://doi.org/10.3390/rs17203470>
- SatLoc (*Remote Sensing* 2025): <https://doi.org/10.3390/rs17173048>
- Bearing-UAV (CVPR 2026): <https://arxiv.org/abs/2603.22153>
- TRAIL (ECCV 2026): <https://arxiv.org/abs/2609.07373>
- Kinnari et al., season-invariant UAV localization (RA-L 2022): <https://arxiv.org/abs/2110.01967>; orthophoto matching (2021): <https://arxiv.org/abs/2103.14381>
- Jurevičius et al., particle filter + VO (*Machine Vision and Applications* 2019): <https://arxiv.org/abs/1910.12121>
- Holistic Fusion, factor-graph state estimation (2025): <https://arxiv.org/abs/2504.06479>
- Altitude-adaptive geo-localization (2026): <https://arxiv.org/abs/2602.23872>
- Scale-aware UAV-to-satellite CVGL (2026): <https://arxiv.org/abs/2603.07535>
- Reference-set fine-tuning for VPR (2025): <https://arxiv.org/abs/2510.03751>
- Aerial localization dataset list: <https://github.com/michaelschleiss/awesome-aerial-localization-datasets>

**Encoders**

- MegaLoc: <https://arxiv.org/abs/2502.17237>
- DINOv3 (models and licence): <https://github.com/facebookresearch/dinov3>, <https://arxiv.org/abs/2508.10104>
- InfoGeo (ICML 2026): <https://arxiv.org/abs/2605.07099>
- EGS: <https://arxiv.org/abs/2509.20684>
- VFM-Loc: <https://arxiv.org/abs/2603.13855>
- CAMP (TGRS 2024): <https://github.com/Mabel0403/CAMP>
- Game4Loc / GTA-UAV (AAAI 2025): <https://github.com/Yux1angJi/GTA-UAV>
- D²-VPR (AAAI 2026): <https://arxiv.org/abs/2511.12528>
- C-RADIOv3: <https://github.com/nvlabs/radio>

**Matching**

- MatchAnything: <https://arxiv.org/abs/2501.07556>
- MINIMA (CVPR 2025): <https://github.com/LSXI7/MINIMA>
- GIM (ICLR 2024): <https://arxiv.org/abs/2402.11095>
- XFeat (CVPR 2024): <https://github.com/verlab/accelerated_features>
- LightGlue ONNX/TensorRT, including Orin Nano timings: <https://github.com/figurerobotics/LightGlue-ONNX>
- STHN: <https://arxiv.org/abs/2405.20470>; UASTHN (ICRA 2025): <https://arxiv.org/abs/2502.01035>

**VO / VIO**

- Jetson VIO benchmark (RA-L 2021): <https://arxiv.org/abs/2103.01655>
- cuVSLAM: <https://arxiv.org/abs/2506.04359>
- AirSLAM (TRO 2025): <https://github.com/sair-lab/AirSLAM>
- LEVIO: <https://arxiv.org/abs/2602.03294>
- RaD-VIO: <https://arxiv.org/abs/1810.08704>
- Monocular SLAM on high-altitude nadir footage (2026): <https://arxiv.org/abs/2608.18632>
- DPVO: <https://arxiv.org/abs/2208.04726>
- PX4 EKF2 tuning: <https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf>

**Hardware**

- Jetson Nano (472 GFLOPS; ResNet-50 at 36 FPS FP16): <https://developer.nvidia.com/blog/jetson-nano-ai-computing/>
- Orin Nano Super: <https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/>
