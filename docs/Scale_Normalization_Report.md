---
title: "Scale Normalisation from Height Above Ground"
subtitle: "Does knowing the drone's altitude make satellite retrieval generalize? — UAV-VisLoc experiment"
date: "2026-10-02"
---

# Scale Normalisation from Height Above Ground

**Question.** Retrieval compares each drone frame with square satellite tiles of a fixed
ground size. When the frame covers more or less ground than a tile, the encoder sees the same
place at the wrong zoom. Can the vehicle's own altitude, turned into height above ground with a
public terrain model, fix this per frame? And can it replace the two things the pipeline uses
today: a tile size tuned by a ground-truth sweep, and a blind three-scale search that triples
the encoder cost?

## Summary

**Yes, when the scales disagree, and the decision can be made from altitude alone.**

1. **Height above ground is cheap and accurate.** Flight-log altitude minus the free Copernicus
   GLO-30 terrain model gives AGL to within a few metres, checked on r06's take-off pad. It
   matters: r05 flies at 2,313 m above sea level but only 374 m above the ground. It also works
   with a navigation prior 300 m off.
2. **One constant per lens turns AGL into ground footprint.** One camera flown over four
   regions at 308–778 m AGL needs a single `k` (53° field of view), and two unrelated encoders
   measure it within 1.2 % of each other. Pixel count does not identify the lens: two regions
   that both shoot 3000 × 2000 px need fields of view of 52° and 62–73°.
3. **Where the map's tiles are the wrong size, this is the biggest single lift measured in the
   project.** On r11, whose map was tiled at an untuned 250 m while each frame covers about
   470 m, MegaLoc goes from **26.4 % to 63.9 %** top-1 within 100 m (median error 165 → 77 m;
   58 frames won, 4 lost, p < 10⁻¹²). That is +9 points over the blind three-scale search, at
   one third of its encoder cost. Sizing the map from AGL instead (470 m tiles, centre-aligned)
   gets to 55.6 %. That is not significantly different from the crop (34 / 22, p = 0.14).
4. **Where the tiles already fit the frames, it adds nothing.** r05 and r06 had been sized by
   trial and error. On r10, whose `k` is the least certain (the two encoders disagree by 21 %),
   cropping with the higher value costs 7.6 points (p = 0.05). With the lower value, the gate
   leaves r10 untouched (§5.7).
5. **The recommended policy is a gate:** crop the frame to the tile's ground size only when AGL
   says it covers more than about 1.25× the tile. Over the four regions this lifts the current
   pipeline's mean from 45.1 % to 52.4–54.5 % for any gate from 1.1 to 1.7. Run end to end
   through the shipped command, it gives **45.1 % → 55.4 %**, and no region does worse
   (§5.7).
6. **AGL also picks the map's tile size without ground truth.** The rule "tile = median ground
   side of the frame" lands within 5 % of the tile sizes earlier found by trial and error
   (r05: 240 vs 250 m; r06: 210 vs 200 m). It corrects r11's untuned default (+29.2 points for
   MegaLoc, +22.2 for DenseUAV-ViT, both p < 0.001, centre-aligned maps). On maps that had
   already been tuned it is about as good, not better. Over all four regions it gains +4.3
   points (MegaLoc, 85 / 60, p = 0.046) and +4.5 points (DenseUAV-ViT, p = 0.009). Undersized
   tiles are the one clear way to lose (−7 to −13 points).
7. **Where the tile grid falls is itself worth ±12 points.** The same map with its grid shifted
   by half a stride moves top-1 within 100 m by up to 12 points on r05. So the absolute numbers
   in this project carry grid luck, and maps can only be compared fairly when their tile centres
   coincide (§5.6). Every query-side result above compares policies on the *same* map and is
   unaffected.

Both pieces are in the main code path: `visloc_eval.py --scale-from-agl --camera-k K --agl-gate 1.25`
and `visloc_scale.py plan --rule short-median`.

---

## 1. Where the height comes from

### 1.1 In UAV-VisLoc

The drone frames carry no EXIF or XMP metadata: no focal length, no relative altitude. This was
checked on regions 01, 05, 06, 10 and 11. The only height is the region CSV's `height` column.
It is the flight-log altitude **above sea level**, held almost constant by the autopilot.

Height above ground (AGL) is that altitude minus the terrain under the vehicle. The terrain
comes from the **Copernicus GLO-30 DEM**: 30 m posting, free, one Cloud-Optimised GeoTIFF per
1° cell on the AWS open-data bucket. Eight cells, about 320 MB, cover all eleven regions.
`avl/terrain.py` downloads them once and samples them bilinearly.

| Region | Camera | Images | Altitude ASL (median) | Terrain under track | AGL min / median / max (whole flight) | AGL, 144 evaluated frames |
|---|---|---:|---:|---:|---:|---:|
| 01 | B | 718 | 406 m | 13–40 m | 365 / 388 / 396 m | 368–396 m |
| 02 | B | 536 | 406 m | 12–47 m | 358 / 390 / 396 m | 368–395 m |
| 03 | B | 384 | 466 m | 0–18 m | 448 / 462 / 467 m | 448–467 m |
| 04 | B | 369 | 544 m | 0–13 m | 532 / 540 / 545 m | 532–545 m |
| **05** | A | 473 | **2,313 m** | 1,911–2,123 m | 190 / **374** / 407 m | 360–404 m |
| **06** | B | 344 | 839 m | 496–602 m | −7 / 308 / 344 m | **−7–344 m** (take-off frames) |
| 08 | B | 517 | 551 m | 0–10 m | 542 / 549 / 552 m | 544–552 m |
| 09 | B | 383 | 546 m | 0–138 m | 408 / 544 / 546 m | 495–546 m |
| **10** | A | 144 | 773 m | 475–488 m | 286 / 294 / 299 m | 286–299 m |
| **11** | B | 295 | **2,573 m** | 1,716–2,010 m | 244 / **778** / 858 m | **244–808 m** |

Several regions ship fewer images than CSV rows (r02 536 of 1,071, r08 517 of 1,033, r11 295 of
590); all numbers here use the images on disk. Camera A is 3000 × 2000 px and camera B is
3976 × 2652 px. These are pixel formats only, not known lenses (§2).

![Flight-log altitude, terrain and AGL along each evaluated flight](report_figs/scale/s1_agl_profiles.png)

Observations:

- **Sea-level altitude is a poor scale proxy.** r05 flies at 2,313 m but only 374 m above the
  ground. r11 flies at 2,573 m, and the terrain beneath it rises and falls by 300 m, so the
  footprint changes by up to 3× within the evaluated frames.
- **Datum check.** r06's CSV contains six frames shot on the take-off pad (identical position,
  earlier timestamps; the CSV is not strictly time-ordered). There the log altitude sits
  **6–7 m below** the GLO-30 surface. GLO-30 is a *surface* model, so trees and buildings raise
  it. The log altitude and the DEM therefore agree to within a few metres at a point of known
  zero AGL: whatever datum the log uses, it does not introduce a tens-of-metres offset here.
- **Within-flight variation** of the footprint (p90/p10 of AGL over the evaluated frames) is
  only 1.07 on r05 and 1.03 on r10, but 1.29 on r06 and 1.21 on r11. Per-frame normalisation
  can only matter where there is per-frame variation.

### 1.2 On the real vehicle (no GNSS)

| Source | Gives | Use |
|---|---|---|
| Barometer (every flight controller) | altitude relative to take-off; ±1–3 m short-term, drifts a few m/h | main source. Add the take-off point's DEM height for ASL, then subtract the DEM at the *estimated* position |
| LiDAR / radar altimeter | AGL directly | ≤ ~40–200 m only (DenseUAV-like altitudes) |
| DJI XMP (`RelativeAltitude`) | barometric altitude per photo | free if the camera is DJI; stripped from this dataset |
| Map match | true scale after a verified match | corrects barometer drift in flight |

The terrain lookup needs a horizontal position, which is what is being estimated. The
navigation prior is enough, because the DEM is smooth at the scale of a prior's error. §5.1
tests this with the DEM sampled 300 m (1σ) away from the truth.

---

## 2. The camera footprint constant *k*

For a nadir pinhole camera, the full frame's ground width is `k · AGL`, with
`k = 2·tan(HFOV/2)`. On a real vehicle, `k` comes from the lens datasheet. UAV-VisLoc does not
document its cameras, so `k` was **measured**:

1. Take a frame and turn it north-up with its logged heading.
2. Cut north-up satellite patches of 11 ground sizes (a 2^(1/3) geometric ladder) centred on
   its logged position.
3. Find the patch the encoder (MegaLoc) finds most similar.

The best size divided by the frame's ground extent gives `k`. Single frames are noisy, so the
per-frame curves (all on the same grid of `k`) are z-scored and averaged, and `k` is the peak of
the mean curve.

Sixteen frames per region were scanned in six regions (96 frames). The scan was repeated with a
second, unrelated encoder (DenseUAV-ViT) as a cross-check. The value *used* for each region is
measured on frames **outside** the 144 evaluated ones whenever the flight is long enough. r10
has only 144 frames, so its `k` necessarily comes from evaluated frames; it is a single constant,
not a per-frame fit.

| Region | Camera format | Median AGL | k, MegaLoc | k, DenseUAV-ViT | Agreement | HFOV implied |
|---|---|---:|---:|---:|---:|---:|
| r01 | B | 388 m | 1.009 | 1.003 | 0.6 % | 53.5° |
| r06 | B | 308 m | 1.050 | 1.015 | 3.4 % | 55.4° |
| r08 | B | 549 m | 0.883 | 1.001 | 13 % | 47–53° |
| r11 | B | 778 m | 0.959 | 0.972 | 1.4 % | 51.2° |
| r05 | A | 374 m | 0.971 | 0.991 | 2.0 % | 51.8° |
| r10 | A | 294 m | 1.476 | 1.197 | **21 %** | 62–73° |
| **camera B pooled** | | | **1.002** | **0.990** | **1.2 %** | **53°** |

![Calibration: per-frame estimates (left) and the per-region constants actually used (right)](report_figs/scale/s2_calibration.png)

![Similarity versus satellite-patch ground size for one frame per region](report_figs/scale/s3_scale_curves.png)

What this shows:

1. **One constant per lens works.** Camera B was flown over four regions at median AGLs from
   308 to 778 m. Its region constants span 0.88–1.05, and two unrelated encoders agree on the
   pooled value within 1.2 %. A 53° horizontal field of view is what a 35 mm-equivalent lens
   gives.
2. **The same pixel count is not the same lens.** r05 and r10 both shoot 3000 × 2000 px, but r10
   needs a much wider field of view (62–73° vs 52°), plausibly a 24 mm-equivalent lens. Pooling
   "camera A" would therefore be wrong by about 40 % for one of them. That is why the main
   protocol uses `k` per region (the stand-in for knowing your own camera) and reports
   leave-region-out only as a stress test (§5.4).
3. **r10's constant is the least certain.** The two encoders disagree by 21 %, and it cannot be
   measured outside the evaluated frames.
4. **Geometry is a first-order model, not the whole story.** For camera B, the *per-frame*
   matched footprint grows as AGL^0.69 (95 % bootstrap CI 0.54–0.83), not AGL^1. Each region
   spans a narrow AGL band, so this slope mixes altitude with scene and imagery differences
   between regions. Within ±10 %, though, the encoder's preferred scale is not purely
   geometric. The retrieval experiment (§5) is the test that counts.

---

## 3. A hidden scale error in exact north-alignment

`rotate_no_padding` keeps the largest square with no padding after rotating the frame, and its
side is `min(w, h) / (|cos θ| + |sin θ|)`. So, before any deliberate scaling, **the ground covered
by the encoded square already depends on the heading**:

| Region | Heading (median) | Square ÷ short side, p10 / p50 / p90 |
|---|---:|---:|
| r05 | ±97° | 0.79 / 0.84 / 0.89 |
| r10 | −29° | 0.71 / 0.76 / 0.94 |
| r06 | −15° | 0.76 / 0.82 / 0.94 |
| r11 | ±90° | 0.94 / 0.98 / 1.00 |

The older "nearest 90° rotation" baseline keeps the full short-side square, so its scale is
constant, but it leaves up to ±45° of residual rotation. Exact north-alignment fixes the
rotation and introduces up to ~30 % frame-to-frame scale jitter. The AGL crop accounts for
this through `avl/scale.py::query_square_side_px`. The shipped `--north-align-mode auto`
rotates exactly only the frames it crops (the crop fits inside the rotated square anyway) and
snaps the others to 90°, keeping their full square.

---

## 4. Protocol

- **Regions and frames.** r05, r10, r06 (the three regions with earlier MegaLoc results) and
  **r11**, which was never tuned or evaluated before. In each, the first 144 frames, exactly
  as in all earlier numbers.
- **Encoders.** MegaLoc (DINOv2-B + SALAD, the recommended generalist) and DenseUAV-ViT
  (UAV-trained, cheap). `k` always comes from the MegaLoc calibration, because it is a camera
  property.
- **Maps.** Each region's existing tile set (r05: 250 m tiles at a **100 m** stride, despite the
  name; r10, r06: 200 m at 100 m; r11: an untuned 250 m default at 125 m), plus **AGL-sized
  maps**. Their tile size is the median ground side of the full frame, from `k` and AGL with no
  ground truth (`plan --rule short-median`), at the same stride and **centre-aligned** with the
  original map so only tile size differs (§5.6 shows why that matters). Deliberately undersized
  maps (r05 190 m, r06 150 m) show where cropping starts to pay.
- **Query-scale policies.** All share the same encoder and map:

| Policy | What the encoder sees | Encodes / frame |
|---|---|---:|
| `rot90` | short-side square, rotated by the 90° step nearest north (the current heading baseline) | 1 |
| `north` | exact north-up square (`rotate_no_padding`), uncropped | 1 |
| `blind3` | `north` at crops 1, 0.79, 0.63; best score per tile wins (today's `--query-scales`) | 3 |
| **`agl`** | `north` cropped so it covers exactly one tile's ground (AGL + `k`) | **1** |
| `agl_prior` | same, but DEM sampled at a prior position N(0, 300 m) away, plus a per-flight barometric bias N(0, 15 m) | 1 |
| `oracle_grid` | per frame, whichever of crops 1 / 0.79 / 0.63 / 0.5 lands closest (**uses ground truth**: an upper bound, not a method) | 4 |

- **Centering.** Off, and `map+flight` (causal flight mean; the most consistent mode measured
  so far).
- **Metrics.** Top-1 within 25 / 50 / 100 / 200 / 500 m, top-5 within 100 m, cluster-fused
  pose within 100 m, and median error. Paired exact McNemar tests on top-1 within 100 m. With 144
  frames, one frame is 0.7 pts, so differences under ~5 pts need the test before they mean
  anything.

---

## 5. Results

### 5.1 Query side: crop the frame to the tile's ground size

MegaLoc, 144 frames per region, top-1 within 100 m (median error in brackets). Each region's
map is the tile set used so far.

**Centering `map+flight`:**

| Region (map) | Frame ÷ tile | 90° view (current) | exact north | blind 3-scale (3×) | **AGL crop (1×)** | AGL crop, DEM at prior | oracle scale (GT) |
|---|---:|---:|---:|---:|---:|---:|---:|
| r05 (250 m) | 0.96 | **74.3** (66 m) | 67.4 | 67.4 | 67.4 *(no crop needed)* | 67.4 | 72.9 |
| r10 (200 m) | 1.45 * | **36.8** (176 m) | 36.8 | 31.9 | 29.2 (397 m) | 31.9 | 46.5 |
| r06 (200 m) | 1.06 | **43.1** (269 m) | 40.3 | 41.7 | 41.7 | 41.7 | 45.1 |
| **r11 (250 m, untuned)** | **1.88** | 26.4 (165 m) | 27.8 | 54.9 (93 m) | **63.9 (77 m)** | 60.4 (83 m) | 73.6 |

**Centering off:**

| Region (map) | 90° view | exact north | blind 3-scale | AGL crop | AGL crop, DEM at prior | oracle |
|---|---:|---:|---:|---:|---:|---:|
| r05 (250 m) | **68.1** | 64.6 | 65.3 | 64.6 | 64.6 | 72.2 |
| r10 (200 m) | **22.9** | 22.9 | 19.4 | 20.1 | 22.2 | 28.5 |
| r06 (200 m) | **43.1** | 39.6 | 38.2 | 37.5 | 38.9 | 43.8 |
| **r11 (250 m)** | 26.4 | 26.4 | 49.3 | **59.0** | 59.0 | 64.6 |

"Frame ÷ tile" is the ground side of the full short-side square predicted from AGL and `k`,
divided by the tile size. The \* marks r10, whose `k` is the uncertain one (§2); with
DenseUAV-ViT's `k` the ratio would be 1.17. The AGL-crop policy is built on the exact-north
view, so where no crop is needed it equals "exact north". The shipped `--north-align-mode auto`
uses the 90° view for uncropped frames instead, which is why §5.7's numbers are higher.

**Paired tests** (exact McNemar, top-1 within 100 m, `map+flight`):

| Comparison | Region | Δ pts | frames won / lost | p |
|---|---|---:|---:|---:|
| AGL crop vs 90° view | **r11** | **+37.5** | **58 / 4** | **< 10⁻¹²** |
| AGL crop vs blind 3-scale | **r11** | **+9.0** | **19 / 6** | **0.015** |
| AGL crop vs uncropped north | r11 | +36.1 | 57 / 5 | < 0.001 |
| AGL crop vs uncropped north | r10 | −7.6 | 8 / 19 | 0.052 |
| AGL crop vs uncropped north | r05, r06 | 0.0, +1.4 | 0 / 0, 3 / 1 | 1.0, 0.63 |
| AGL crop vs uncropped north | **pooled, 4 regions** | **+7.5** | **68 / 25** | **< 0.001** |
| DEM at prior (±300 m) + baro bias vs DEM at truth | r11 | −3.5 | 5 / 10 | 0.30 |
| exact north vs 90° view | pooled, 4 regions | −2.1 | 38 / 50 | 0.24 (and flips with grid placement, §5.6) |

**Reading.**

- **When the frame and tile disagree in scale, the AGL crop is the single largest lift measured
  in this project.** On r11, MegaLoc goes from 26.4 % to 63.9 % (median error 165 → 77 m). That
  beats the blind three-scale search by 9 points at a third of the encoder cost. The literature
  figure for altitude-based scale normalisation was +41.5 pts R@1; this experiment measures
  +37.5.
- **When they already agree, it does nothing.** On r05 and r06 the earlier maps had been sized
  to the frames by trial and error. The crop is then a no-op (r05: 100 % of frames need no
  crop) or a wash.
- **r10 is the one loss.** The loss sits in the strongly cropped frames (32 → 15 % for crop
  fractions below 0.8), and r10 is where `k` is least certain: MegaLoc's scan says the frame is
  1.45× the tile, DenseUAV-ViT's says 1.17×. Either the crop removes context MegaLoc needed, or
  `k` is too high and the crop overshoots. Maps sized for both values (§5.4) do not settle it.
- **The DEM does not need the true position.** Sampling the terrain 300 m (1σ) from the
  truth, plus a 15 m barometric bias, changes r11 by −3.5 pts (not significant) and changes
  nothing in the other regions.

![Per-frame top-1 error against AGL on r11: uncropped north (grey) vs AGL crop (blue)](report_figs/scale/s5_error_vs_agl_r11.png)

### 5.2 How wrong can the height be?

The r11 frames were rerun with the AGL deliberately scaled by −20 / −10 / +10 / +20 %:

| AGL error | −20 % | −10 % | 0 | +10 % | +20 % | no crop |
|---|---:|---:|---:|---:|---:|---:|
| top-1 within 100 m | 53.5 | 61.8 | **63.9** | 63.2 | 58.3 | 27.8 |

![Sensitivity of the AGL crop to height error (r11)](report_figs/scale/s6_sensitivity_megaloc.png)

The optimum is flat within ±10 %. At ±20 % the crop still keeps more than 80 % of its gain.
Barometer drift, the DEM's ~4 m vertical accuracy, a wrong datum, or an 8 % error in `k` all
fall well inside this band.

### 5.3 When to crop: a gated policy

The crop helps when the frame is much larger than a tile and costs context when it is not. So
the obvious rule is to crop **only when the AGL-predicted frame ÷ tile ratio exceeds τ**, and
otherwise keep the full 90° view. This was computed from the saved per-frame results with no
re-encoding, over a range of τ rather than a value tuned on these frames:

| Region (map so far) | Frame ÷ tile | 90° view (current) | always crop | τ = 1.1 | τ = 1.2 | τ = 1.3 | τ = 1.5 | τ = 1.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r05 (250 m) | 0.96 | 74.3 | 67.4 | 74.3 | 74.3 | 74.3 | 74.3 | 74.3 |
| r10 (200 m) | 1.45 \* | 36.8 | 29.2 | 29.2 | 29.2 | 29.2 | 36.8 | 36.8 |
| r06 (200 m) | 1.06 | 43.1 | 41.7 | 42.4 | 44.4 | 43.1 | 43.1 | 43.1 |
| r11 (250 m) | 1.88 | 26.4 | 63.9 | 63.9 | 63.9 | 63.9 | 63.9 | 59.7 |
| **mean of the four** | | **45.1** | 50.5 | **52.4** | **53.0** | **52.6** | **54.5** | **53.5** |

*(MegaLoc, `map+flight`, top-1 within 100 m.)*

**Any τ between 1.1 and 1.7 lifts the four-region mean by 7–9 points over the current pipeline.**
The gate keeps all of r11's gain and none of the cropping losses, except on r10. There
MegaLoc's `k` makes the frame look 1.45× the tile, so any τ below 1.45 crops it. With
DenseUAV-ViT's `k` the ratio is 1.17, and τ = 1.25 leaves it alone (§5.7).

Where the benefit starts can be read off maps deliberately cut too small, each compared with
itself (same map, so no grid noise):

| Map | Frame ÷ tile | 90° view | AGL crop | Δ |
|---|---:|---:|---:|---:|
| r05, 190 m tiles | 1.26 | 66.7 | 65.3 | −1.4 |
| r06, 150 m tiles | 1.41 | 30.6 | 38.2 | **+7.6** |
| r11, 250 m tiles | 1.88 | 26.4 | 63.9 | **+37.5** |

The crop starts paying somewhere between 1.26× and 1.41× and grows quickly beyond that. The gate
is implemented as `avl/scale.py::query_scale(..., gate=τ)` and `visloc_eval.py --agl-gate τ`.
**The shipped default is τ = 1.25**, validated end to end in §5.7. A τ of 1.3–1.5 would be
marginally safer near the onset; that was not tuned here, to avoid fitting τ to these frames.

### 5.4 Map side: size the tiles from altitude instead

The alternative is to leave the frame whole and build the map at the frame's scale. The tile is
the median ground side of the full short-side square, from `k` and the AGL along the planned
flight (`visloc_scale.py plan --rule short-median`), and **no ground truth is involved**. Every
map below has the original map's stride and is **centre-aligned** with it. Its tile centres sit
on exactly the same points (the median distance to the nearest centre is identical), so only
the tile size differs (§5.6 explains why that matters).

MegaLoc, 90° view, top-1 within 100 m. Brackets: Δ vs the original map (frames won / lost, p).

| Region | Map | Tile | `map+flight` | off |
|---|---|---:|---:|---:|
| r05 | original (best of 250/400/600 tried) | 250 m | 74.3 | 68.1 |
| r05 | **AGL-sized** | **240 m** | 67.4 (−6.9; 5 / 15, 0.04) | 67.4 (−0.7; 7 / 8, 1.0) |
| r05 | undersized | 190 m | 66.7 (−7.6; 13 / 24, 0.10) | 59.7 (−8.3; 19 / 31, 0.12) |
| r06 | original (never swept) | 200 m | 43.1 | 43.1 |
| r06 | **AGL-sized** | **210 m** | 44.4 (+1.4; 10 / 8, 0.81) | 41.7 (−1.4; 6 / 8, 0.79) |
| r06 | undersized | 150 m | 31.9 (**−11.1**; 5 / 21, 0.002) | 33.3 (**−9.7**; 6 / 20, 0.01) |
| r10 | original (best of a 100–600 m sweep) | 200 m | 36.8 | 22.9 |
| r10 | AGL-sized, DenseUAV-ViT's `k` | 235 m | 31.2 (−5.6; 14 / 22, 0.24) | 22.2 (−0.7; 9 / 10, 1.0) |
| r10 | AGL-sized, MegaLoc's `k` | 290 m | 30.6 (−6.2; 13 / 22, 0.18) | 29.2 (+6.3; 19 / 10, 0.14) |
| **r11** | original (untuned default) | 250 m | 26.4 | 26.4 |
| **r11** | **AGL-sized** | **470 m** | **55.6 (+29.2; 57 / 15, < 0.001)** | **52.8 (+26.4; 56 / 18, < 0.001)** |

- **Where the map had already been tuned, the AGL-sized map is about as good.** The rule
  lands within 4–5 % of the size found by trial and error (r05 240 vs 250 m, r06 210 vs 200 m).
  Of the six AGL-sized comparisons, five are within noise. One (r05, `map+flight`) is 6.9 points
  worse at p = 0.04, which is not significant once six comparisons are counted. So: no labelled
  sweep is needed to get a usable map, but it does not beat a map that was already tuned.
- **Undersized maps clearly hurt.** Tiles 25–30 % smaller than the frame lose 8–11 points
  (r06 at 150 m: p ≤ 0.01). Erring large is safer than erring small.
- **r10 stays open.** 290 m beats 200 m without centering (+6.3) and loses with it (−6.2),
  neither significant. Its `k` cannot be pinned down from this data.
- **Where the default was wrong, AGL sizing is the fix.** r11 gains 29.2 points with the tile
  centres held fixed. On the same frames, the query-side crop reaches 63.9 %, not significantly
  different (34 / 22, p = 0.14). Pooled over the four regions, AGL-sized maps gain +4.3 points
  (85 / 60, p = 0.046).
- **Per-frame map selection** was also implemented: each frame is searched in whichever map's
  tile is nearest its AGL-predicted footprint (`map_select` in the eval JSON). Within the
  evaluated frames the AGL varies too little for this to matter: it collapses to essentially one
  map per region. It becomes relevant for flights whose altitude or terrain changes by more
  than ~1.3× (take-off and climb-out, mountain ridges).

![Query side vs map side, all regions (MegaLoc, map+flight)](report_figs/scale/s7_headline_megaloc_map_flight.png)

### 5.5 Replication with a second encoder (DenseUAV-ViT)

DenseUAV-ViT is a small (4.2 GMAC) model trained on low-altitude UAV↔satellite pairs. It is much
weaker on these regions, but it answers whether the effect belongs to MegaLoc or to the problem.
Same frames, same `k` (from the MegaLoc calibration), same maps; `map+flight` centering, top-1
within 100 m:

| Region | 90° view, map so far | AGL crop (query side) | blind 3-scale | 90° view, AGL-sized map (centre-aligned) | undersized map |
|---|---:|---:|---:|---:|---:|
| r05 | 39.6 | 38.9 | 32.6 | 37.5 (240 m; −2.1, p = 0.55) | 29.9 (190 m; **−9.7**, p = 0.01) |
| r10 | 21.5 | 19.4 | 18.1 | 17.4 (290 m; −4.2, p = 0.38) | — |
| r06 | 22.2 | 25.7 | 22.2 | 24.3 (210 m; +2.1, p = 0.45) | 15.3 (150 m; **−6.9**, p = 0.03) |
| **r11** | 16.0 | 22.9 | 25.0 | **38.2** (470 m; **+22.2**, 37 / 5, p < 0.001) | — |
| pooled Δ vs 90° view (wins / losses, p) | | +1.9 (28 / 17, 0.14) | −0.3 (n.s.) | **+4.5 (59 / 33, 0.009)** | |

- **r11's scale mismatch hurts both encoders, and AGL rescues both.** With centre-aligned maps
  DenseUAV-ViT gains +22.2 points (37 / 5, p < 0.001) and MegaLoc +29.2 (57 / 15,
  p < 0.001). As with MegaLoc, AGL-sized maps elsewhere are within noise of the tuned ones, and
  undersized maps hurt (−7 to −10 points, p ≤ 0.03).
- **Which side to normalise on depends on the encoder.** MegaLoc does about as well with the
  cropped frame (63.9 %) as with the AGL-sized map (55.6 %; p = 0.14). DenseUAV-ViT gains far more from the
  AGL-sized map (38.2 %) than from the crop (22.9 %): it loses more by having context cut away.
  **Map-side sizing is therefore the encoder-robust choice,** and the crop is the in-flight
  fallback for when the altitude leaves the planned band.

### 5.6 Control: how much is grid luck?

The map was rebuilt with **identical tile size and stride, but the grid origin shifted by half a
stride** (50 m). Nothing about scale changes, only where the tile centres fall relative to the
flight line. MegaLoc, top-1 within 100 m:

| Map | Nearest tile centre (median) | 90° view, `map+flight` | 90° view, off | exact north, `map+flight` |
|---|---:|---:|---:|---:|
| r05, original grid | 33.4 m | 74.3 | 68.1 | 67.4 |
| r05, grid shifted 50 m | 40.2 m | **62.5** | **56.9** | 68.1 |
| r06, original grid | 41.5 m | 43.1 | 43.1 | 40.3 |
| r06, grid shifted 50 m | 40.2 m | 45.8 | 43.8 | 45.8 |

**Grid placement alone moves results by up to 12 points.** r05 flies a straight east–west line,
and its original grid happens to put a row of tile centres close to the track (top-1 within
50 m: 45.8 % vs 19.4 % on the shifted grid). Three consequences:

1. **Comparisons on the same map are safe.** Every query-side result in §5.1–5.3 (crop vs no
   crop, gate, blind search) compares policies on one fixed map with paired frames.
2. **Comparisons between differently sized maps are not, unless their tile centres coincide.**
   A different tile size moves the centres (by `(T₁ − T₂)/2`). §5.4's map-side numbers for r05,
   r06 and r10 were therefore re-run with **centre-aligned** maps (offset
   `= (T_old − T_new)/2 mod stride`, `visloc_prepare.py --offset-m`). r11 was re-run the same
   way: +32.6 points before alignment, +29.2 after, far outside the noise either way.
3. **The earlier "the 90° view beats exact north-alignment" finding does not survive.** On the
   shifted r05 grid the order reverses (62.5 vs 68.1). It is withdrawn; the two are equivalent
   within grid noise.

A practical corollary for the real system: a denser stride, or a final position refined by
matching or by the trajectory filter, removes most of this luck. The 100–125 m strides used
here quantise positions on the scale of the 100 m success threshold itself.

### 5.7 The shipped command reproduces the experiment

The production path was run end to end, with nothing from the experiment harness, on the four
original maps:

```bash
python scripts/visloc_eval.py --refs <map>/references.csv --queries <map>/queries.csv \
  --model megaloc --rotations 1 --north-align --yaw-sign -1 --north-align-mode auto \
  --scale-from-agl --camera-k <k> --agl-gate 1.25 --center map+flight --max-queries 144
```

| Region | `k` | Frames cropped | Shipped command | Experiment (gated τ = 1.25) | Current pipeline (90° view) |
|---|---:|---:|---:|---:|---:|
| r11 | 0.959 | 99 % | **65.3** | 63.9 | 26.4 |
| r05 | 0.971 | 0 % | 74.3 | 74.3 | 74.3 |
| r06 | 1.050 | 0 % | 44.4 | 43.1 | 43.1 |
| r10 | 1.197 (DenseUAV-ViT's) | 0 % | 37.5 | 36.8 | 36.8 |
| **mean** | | | **55.4** | | **45.1** |

The shipped command matches the experiment within 1–2 frames per region. The differences come
from the 2-pixel border `rotate_no_padding` keeps and from the take-off frame the gate leaves
uncropped. **The gated command adds 10.3 points to the four-region mean and never does worse than
the current pipeline.** With DenseUAV-ViT's `k` for r10 (1.20 instead of 1.48), the gate
correctly leaves r10 alone. With MegaLoc's it would have cropped and lost 7.6 points, which is
the practical argument for getting `k` from the lens datasheet.

---

## 6. Recommendation for the vehicle

1. **Always compute AGL:** barometric altitude (plus the take-off point's terrain height) minus
   the GLO-30 terrain at the navigation prior. Cost: microseconds and ~0.2 MB of terrain for a
   city.
2. **Know `k` for your lens**, from its datasheet: `k = 2·tan(HFOV/2)`. If it is in doubt,
   measure it with `visloc_scale.py calibrate` on a dozen frames over known ground, and check it
   with two encoders as done here. A wrong `k` is the one way this was seen to cost accuracy.
3. **Build the map at the planned flight's scale:** tile = the frame's ground side at the planned
   AGL (`plan --rule short-median`). If unsure, err large, never small. This is the
   encoder-robust step (§5.5).
4. **In flight, gate-crop:** if the AGL-predicted footprint exceeds the tile by more than 1.25×
   (the vehicle climbed, or the terrain dropped away), crop the frame to the tile's ground size
   (`--scale-from-agl --agl-gate 1.25 --north-align-mode auto`). Otherwise keep the whole frame.
   This is what turned r11 from 26 % into 65 %.
5. **Drop the blind three-scale search.** It never beat the AGL crop and costs 3× the encodes.
6. Treat absolute accuracy figures with care. Grid placement alone is worth ±12 points at a
   100–125 m stride, and the trajectory filter or a matcher should take the last 50 m (see the
   SOTA survey's roadmap).

---

## 7. Cost on the Jetson

| Item | Cost | Notes |
|---|---|---|
| Terrain lookup per frame | microseconds | bilinear read from an in-memory array |
| Terrain data | 52 MB per 1° cell in RAM; ~0.2 MB for an 18 km² city crop | crop the cell to the mission area before flight |
| AGL crop vs blind 3-scale search | **1 vs 3 encodes per frame** | MegaLoc ≈ 53 vs 160 ms on an Orin Nano Super, ≈ 0.5 vs 1.6 s on the original Nano (projections, see the SOTA survey §9) |
| Camera constant `k` | once per lens | datasheet HFOV, or the 16-frame scan used here |
| AGL-sized map | zero at run time | chosen when the map is built, from the planned flight altitude |
| Per-frame map selection | zero encodes; index memory × number of tile levels | each level is one flat index |

The AGL crop is a rare change that is both more accurate and cheaper than what it replaces.

---

## 8. Limitations

- **Sample size.** 144 frames per region and four regions. Only one region (r11) has a large
  frame/tile mismatch, and its result is very strong (p < 10⁻¹²), but it is one region. The
  next regions to add are those whose terrain changes the AGL a lot: r09 (terrain 0–138 m)
  and the full r05 flight (190–407 m).
- **`k` was measured with ground-truth positions,** as a stand-in for the lens datasheet that
  UAV-VisLoc does not provide. It used frames outside the evaluated set wherever the flight was
  long enough. On a real vehicle `k` comes from the camera's specification.
- **Nadir assumption.** The CSV's `Omega` reaches 13° on some frames. Off-nadir pointing shifts
  and stretches the footprint, and the attitude-based rectification from the survey (§5.1 step 3)
  was not applied.
- **GLO-30 is a surface model.** Over forest or a city it reads the canopy or roof tops, so AGL
  is biased low by 10–30 m there. That is a 3–10 % scale error at 300 m, inside the tolerance
  measured in §5.2.
- **Encoder dependence.** MegaLoc gains as much from the crop as from an AGL-sized map;
  DenseUAV-ViT gains far more from the map. Any new encoder should be checked both ways before
  relying on the crop alone.
- **Grid luck.** With 100–125 m strides, where the tile centres fall relative to the flight line
  moves results by up to 12 points (§5.6). All paired comparisons here are on the same map or on
  centre-aligned maps. Absolute numbers from different maps, including those elsewhere in this
  project, should be read with that in mind.
- **r10's `k` is unresolved.** The two encoders disagree, and maps sized for both values do not
  separate them.

---

## 9. What changed in the code

| File | Change |
|---|---|
| `avl/terrain.py` (new) | Copernicus GLO-30 download and bilinear sampling; `TerrainModel.agl()`. Handles both GeoTIFF pixel conventions; NaN outside the downloaded cells |
| `avl/scale.py` (new) | `CameraFootprint` (k, HFOV, GSD), `query_square_side_px` (the heading-dependent square), `frame_ground_m`, `crop_fraction`, `query_scale(..., gate=τ)` (the per-frame `scales` tuple) |
| `scripts/visloc_eval.py` | `--scale-from-agl --camera-k K [--agl-gate τ] [--tile-m] [--height-col] [--dem-dir] [--min-agl]`: one informed crop per frame instead of `--query-scales`. `--north-align-mode exact\|snap90\|auto`. With `--prior-sigma`, the DEM is read at the simulated prior position. Recorded in the summary JSON as `scale_from_agl` |
| `scripts/visloc_prepare.py` | `--offset-m`: shift the tile grid (for centre-aligned maps and the grid-luck control) |
| `scripts/visloc_scale.py` (new) | `agl`, `calibrate`, `calsummary`, `plan` (`--rule short-median\|north-p10`), `eval` (`--tag`): this experiment, resumable, with descriptor caching and reuse of identical crops |
| `scripts/visloc_scale_report.py` (new) | figures and tables for this report, including paired and pooled McNemar tests, the gate sweep and the centre-aligned comparison |
| `tests/test_scale.py` (new) | 14 tests: footprint maths, crop vs the real `rotate_no_padding`, gate behaviour, DEM sampling on synthetic GeoTIFFs (pixel-is-point and pixel-is-area) |
| `pyproject.toml` | optional extra `[terrain]` = tifffile + imagecodecs |
| `README.md` | "Scale from altitude (AGL)" section |

The default behaviour of every existing command is unchanged. Raw outputs are in
`artifacts/visloc/scale/`: per-frame CSVs, JSON summaries, `tables.md`, the calibration and the
run logs.

**Reproduce:**

```bash
pip install -e '.[terrain]'
python scripts/visloc_scale.py agl
python scripts/visloc_scale.py calibrate --regions 05 10 06 11 01 08 --frames 16
python scripts/visloc_scale.py plan --regions 11 --base-tiles r11_t250 --strides 125 --prepare
python scripts/visloc_scale.py eval --region 11 --tiles r11_t250 r11_t470_s125 --model megaloc
python scripts/visloc_scale_report.py
# the shipped path, in the normal benchmark:
python scripts/visloc_eval.py --refs data/visloc_avl/r11_t250/references.csv \
       --queries data/visloc_avl/r11_t250/queries.csv --model megaloc --rotations 1 \
       --north-align --yaw-sign -1 --north-align-mode auto \
       --scale-from-agl --camera-k 0.959 --agl-gate 1.25 --center map+flight --max-queries 144
```
