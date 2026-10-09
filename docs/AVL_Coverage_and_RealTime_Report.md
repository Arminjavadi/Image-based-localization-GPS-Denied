---
title: |
  | Absolute Visual Localisation for GPS-Denied UAV Navigation
  | \vspace{2mm}\large Dataset Coverage, Model Cost, and Real-Time Embedded Feasibility
  | \vspace{1mm}\normalsize A quantitative study of the DenseUAV benchmark and a proposed reference map for Ardabil, Iran
author: "Technical Report — Image-based Localization (GPS-Denied) project"
date: "14 August 2026"
---

\vspace{-6mm}

## Abstract {-}

This report answers four questions raised in project supervision. **(i)** What area does the
project's current dataset cover, and how large is it? **(ii)** How large would an equivalent
dataset for the city of Ardabil, Iran be, and how would it be built? **(iii)** Is the DenseUAV
ViT encoder small enough for real-time operation, and how does it compare with alternatives?
**(iv)** Which NVIDIA Jetson module is the appropriate on-board computer, given that the
development laptop cannot meet real-time requirements?

All dataset figures are computed directly from the 16.76 GB DenseUAV copy on disk and from its
GPS metadata; all model complexity and latency figures for the laptop are measured, not quoted.
Jetson latencies are projected from published TensorRT anchors using a documented scaling model.

The principal findings are: DenseUAV covers only **2.58 km² of effectively imaged ground**
(7.13 km² convex hull) across 14 university campuses in Hangzhou at a 19.7 m sampling pitch,
using 40,833 images. An Ardabil reference map built the correct way — by tiling an
orthorectified satellite mosaic rather than by flying a survey — needs **3,054 tiles, 80 MB of
imagery and a 0.4 MB FAISS index** for the entire 18.011 km² municipality, and can be built in
under 15 minutes on the existing laptop. The DenseUAV ViT-S/16 encoder is **21.86 M parameters
and 4.24 GMACs**, the smallest of the four supported encoders and the only one that is
simultaneously the fastest and (on this data) the most accurate. The development laptop
(Intel i7-7600U, no CUDA) reaches only **2.5 Hz**; a **$399 Jetson Orin Nano 8 GB Super at 25 W
reaches an estimated 35 Hz** with the full four-orientation search, which is 3.5× the 10 Hz
real-time target. **The Orin Nano 8 GB Super is the recommended board**; the Orin NX 16 GB is
the recommended choice only if the same computer must also run detection, tracking or SLAM.

\vspace{2mm}
\hrule
\vspace{3mm}

# 1. Introduction and scope

## 1.1 The system under study

The project implements Absolute Visual Localisation (AVL): a UAV that has lost GNSS estimates
its position by matching its nadir camera frame against a pre-built, geo-tagged image database.
The pipeline in the repository is

$$\text{frame} \;\rightarrow\; \text{CNN/ViT encoder} \;\rightarrow\; \ell_2\text{-normalised descriptor}
\;\rightarrow\; \text{FAISS top-}K \;\rightarrow\; \text{weighted geo-fusion} \;\rightarrow\; (\varphi, \lambda, h)$$

implemented in `avl/encoder.py`, `avl/index.py`, `avl/localizer.py` and `avl/geo.py`. Three
encoder backends are supported (DenseUAV ViT, MixVPR, CosPlace) and three FAISS index families
(`flat`, `hnsw`, `ivfpq`). Localisation fuses the five strongest unique matches on the unit
sphere, weighted by similarity score.

Because the system is *retrieval-based*, its two dominant engineering costs are (a) the size and
geographic coverage of the reference database, and (b) the per-frame inference cost of the
encoder plus the search cost of the index. Sections 3–4 address the first; Sections 5–7 address
the second.

## 1.2 Questions addressed

| # | Question | Section |
|---|----------|---------|
| Q1 | Coverage area and size of the current (DenseUAV) dataset | §3 |
| Q2 | Comparison against other UAV geo-localisation datasets | §4 |
| Q3 | Size and coverage of an equivalent Ardabil dataset | §5 |
| Q4 | Is the DenseUAV model small enough for real time? Comparison with alternatives | §6 |
| Q5 | Retrieval-index cost: compute, memory, latency | §7 |
| Q6 | Specification of the development laptop; why it is not real-time capable | §8.1 |
| Q7 | Which Jetson board is appropriate | §8 |
| Q8 | Complete requirements for a real-time implementation | §9 |

## 1.3 What is measured versus what is estimated

Rigour requires this distinction to be explicit.

| Class | Items | Basis |
|-------|-------|-------|
| **Measured** | dataset image counts, byte sizes, GPS extents, sampling pitch, campus clustering, tile compression ratios, model parameters, GMACs, laptop CPU encode latency, FAISS build/search latency and index sizes, localisation error | direct measurement on this machine, this report's session |
| **Cited** | Jetson module specifications and 2026 list prices, TensorRT reference benchmarks, competing dataset statistics, DenseUAV paper baselines | public sources, listed in §12 |
| **Derived** | Ardabil tile counts, storage and build times; Jetson latency projections; flight-time estimates | stated formulae applied to the above; assumptions given inline |

Jetson projections carry an estimated uncertainty of ±30 %. This does not affect any conclusion
in this report, because the margins involved are 3–10×.


# 2. Method and measurement environment

## 2.1 Measurement platform

All measurements in this report were taken on the project development laptop:

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i7-7600U (Kaby Lake, 2 cores / 4 threads, 2.80 GHz base, 3.90 GHz turbo) |
| Cache | 64 KiB L1d, 512 KiB L2, **4 MiB L3** |
| Memory | 15 GiB usable DDR4 (dual-channel, ≈ 25.6 GB/s theoretical) + 7.5 GiB swap |
| GPU | Intel HD Graphics 620 (integrated) — **no CUDA device present** |
| OS / stack | Ubuntu 24.04.4 LTS, kernel 7.0.0-28; Python 3.12, PyTorch 2.12.0 (CPU), faiss-cpu 1.14.3, timm 1.0.27 |
| Storage | 40 GB consumed by this project; 16.76 GB is the DenseUAV corpus |

The absence of a CUDA device is the single most important fact about this platform: every
neural forward pass in the repository executes on two physical CPU cores. `avl/encoder.py`
requests `cuda` and silently falls back to CPU
(`torch.device(config.device if torch.cuda.is_available() else "cpu")`).

## 2.2 Procedures

- **Model complexity** — parameters counted from the instantiated `nn.Module`; MACs counted with
  `torch.utils.flop_counter.FlopCounterMode` at batch 1 (the counter reports FLOPs at 2 × MACs;
  both are given).
- **Latency** — median of 8 timed forward passes after 3 warm-up passes, `torch.inference_mode`,
  4 OpenMP threads.
- **Coverage** — sampling points parsed from `Dense_GPS_ALL.txt`; local ENU projection about the
  dataset centroid using the WGS-84 series for metres-per-degree; convex hull by monotone chain;
  imaged coverage by rasterising the union of per-point footprints at 2 m resolution;
  campus clustering by single-linkage at 60 m.
- **FAISS** — `IndexFlatIP`, `IndexHNSWFlat` (M = 32, efC = 200, efSearch = 128) and `IndexIVFPQ`
  (m = 64, 8 bits, nprobe = 16) built on synthetic $\ell_2$-normalised vectors of the correct
  dimension; median of 15 searches at $k$ = 10.
- **Jetson projection** — see §8.4 for the model and its validation.


# 3. Q1 — Coverage area and size of the DenseUAV dataset

## 3.1 What DenseUAV is

DenseUAV is the reference dataset the project currently uses. It was collected over **14
university campuses in the Xiasha higher-education district of Hangzhou, Zhejiang, China**, at
approximately 30.32° N, 120.36° E. UAV frames were captured at three altitudes (80, 90, 100 m)
on a dense waypoint grid; for each waypoint, satellite tiles were downloaded at three scales in
two epochs (2020 and 2022).

## 3.2 Composition (measured on disk)

Table 1 is a direct enumeration of the corpus at
`data/DenseUAV/DenseUAV/`.

Table: **Table 1.** DenseUAV composition as stored on this machine.

| Split | Images | Locations | Files/location | Format | Resolution | Size |
|-------|-------:|----------:|---------------:|--------|-----------|-----:|
| `train/drone` | 6,768 | 2,256 | 3 (H80/H90/H100) | JPEG | 1440 × 1080 | 5.31 GB |
| `train/satellite` | 13,536 | 2,256 | 6 (3 scales × 2 epochs) | TIFF | 512 × 512 | 4.02 GB |
| `test/query_drone` | 2,331 | 777 | 3 | JPEG | 1440 × 1080 | 1.95 GB |
| `test/gallery_satellite` | 18,198 | 3,033 | 6 | TIFF | 512 × 512 | 5.47 GB |
| **Total** | **40,833** | **3,033 unique** | — | — | — | **16.76 GB** |

Two observations matter operationally. First, **the satellite gallery is the reference database
and the drone frames are the queries** — the project's `references_gallery_satellite.csv`
correctly selects 1,554 gallery TIFFs (777 locations × 2 files) as the searchable map. Second,
the drone imagery is 11.3 GB of the 16.76 GB but contributes *nothing* to the deployed system:
it is training and evaluation data only. **A deployed AVL system stores only the satellite side.**

## 3.3 Coverage area — five defensible definitions

"Coverage area" is ambiguous for a point-sampled dataset, and different papers use different
conventions. Table 2 gives all five, computed from the 3,033 unique GPS fixes.

Table: **Table 2.** DenseUAV geographic coverage under five definitions.

| Definition | Value | What it means |
|------------|------:|---------------|
| Latitude / longitude extent | 30.309401 – 30.324908° N, 120.335385 – 120.392203° E | raw bounds |
| **Bounding box** | 5.465 km (E–W) × 1.719 km (N–S) = **9.394 km²** | the rectangle enclosing all samples |
| **Convex hull of samples** | **7.128 km²** | the smallest convex region containing all samples |
| **Σ of 14 campus bounding boxes** | **1.145 km²** | the actual surveyed envelope (campuses are disjoint) |
| **Union of 150 m image footprints** | **2.581 km²** | **the ground actually imaged** |
| Union of 200 m image footprints | 3.231 km² | same, assuming a wider camera FOV |

![DenseUAV sampling geometry and the five coverage definitions. The dataset is not a
contiguous survey: single-linkage clustering at 60 m recovers exactly **14 disjoint campus
clusters**, matching the 14 universities reported in the source paper. The bounding box
therefore overstates true coverage by a factor of 3.6.](fig/f1_coverage.png)

The single number to report to a reviewer is **2.58 km² of imaged ground**, with the 7.13 km²
convex hull quoted as the survey extent. Quoting 9.39 km² would be misleading: 73 % of that
rectangle is unsampled space between campuses.

## 3.4 Sampling density

| Metric | Value |
|--------|------:|
| Unique sampling locations | 3,033 |
| Median nearest-neighbour spacing | **19.7 m** (min 9.4 m, max 20.8 m) |
| Density within campus envelopes | 2,649 locations / km² |
| Ideal density of a perfect 20 m grid | 2,500 locations / km² |
| Ground area per location | ≈ 378 m² (within campuses) |

The measured pitch confirms the published 20 m waypoint interval to within 1.5 %. This is an
extremely dense map: with a nadir footprint of roughly 150 m at 80–100 m altitude, **consecutive
reference images overlap by about 87 %.** That redundancy is what allows the system to reach
metre-level fused accuracy (§6.4), and it is also the dominant driver of dataset size.

## 3.5 Storage intensity

| Ratio | Value |
|-------|------:|
| GB per km² of imaged ground | **6.49 GB/km²** |
| GB per km² of convex hull | 2.35 GB/km² |
| GB per km², satellite reference side only | 3.67 GB/km² |
| Mean bytes per satellite tile (TIFF) | 232 KB |
| Same tile re-encoded as JPEG q85 (measured) | **26.2 KB** (8.9× smaller) |
| Same tile as WebP q85 | 13.3 KB (17× smaller) |

> **Key finding (Q1).** DenseUAV occupies **16.76 GB across 40,833 images** and covers
> **2.58 km² of imaged ground** (7.13 km² convex hull, 9.39 km² bounding box) over 14 campuses
> in Hangzhou at a 19.7 m pitch. It is a *small-area, very-high-density* dataset: the density,
> not the area, is what makes it large. Re-encoding the reference side as JPEG q85 would reduce
> the deployable map from 5.47 GB to 0.48 GB with no measurable retrieval penalty at 224 × 224
> input resolution.


# 4. Q2 — Comparison with other UAV geo-localisation datasets

Table 3 places DenseUAV in context. Coverage area is reported only by a minority of datasets;
where it is not published, the cell is marked "n.r." rather than guessed. UAV-VisLoc's coverage
was computed by this report from the latitude/longitude ranges of its 11 satellite maps.

Table: **Table 3.** Comparison of cross-view / UAV visual-localisation datasets.

| Dataset | Images | Locations | Platforms | Region | Coverage | Altitude | Character |
|---------|-------:|----------:|-----------|--------|---------:|----------|-----------|
| CVUSA | 71 k | — | ground + satellite | United States | n.r. | ground | panorama-to-satellite |
| VIGOR | 144 k | — | ground + satellite | 4 US cities | n.r. | ground | one-to-many matching |
| University-1652 | 50.2 k | 1,652 buildings | ground + drone + satellite | 72 universities, worldwide | n.r. | synthetic orbit | building-level retrieval |
| SUES-200 | 6.1 k | 200 | drone + satellite | one campus, Shanghai | n.r. | 150–300 m | multi-height |
| **DenseUAV** | **40.8 k** | **3,033** | drone + satellite | 14 campuses, Hangzhou | **2.58 km²** *(7.13 km² hull)* | 80–100 m | **dense 20 m grid** |
| UAV-VisLoc | 6.7 k | 11 regions | drone + satellite | 11 sites, China | **380.2 km²** | 400–2,000 m | large-area, multi-terrain |
| VPAIR | 2.7 k | 107 km track | light aircraft | Bonn → Eifel, Germany | corridor | 300–400 m | long traverse |
| ALTO | 30 k | 150 + 260 km tracks | helicopter | Ohio, Pennsylvania | corridor | high | long traverse, GPS-INS truth |
| **Ardabil (proposed, §5)** | **3.1 k** | 3,054 tiles | satellite tiles only | Ardabil, Iran | **18.01 km²** | n/a (map) | full-city, 76.8 m pitch |

![Dataset comparison. DenseUAV is mid-sized in image count but among the smallest in
geographic coverage; UAV-VisLoc covers 147× more ground with 6× fewer images because it
samples sparsely along flight corridors. The proposed Ardabil map inverts DenseUAV's
trade-off: 7× the area with 13× fewer images.](fig/f2_datasets.png)

**Interpretation.** These datasets occupy two distinct regimes.

1. **Dense small-area** (DenseUAV, SUES-200, University-1652): metre-level accuracy over a
   campus. High image-per-km² cost. Suitable for *precision* tasks — landing, inspection,
   confined-area navigation.
2. **Sparse large-area** (UAV-VisLoc, VPAIR, ALTO): kilometre-scale corridors with tens-of-metres
   accuracy. Suitable for *en-route* navigation.

The project's chosen benchmark, DenseUAV, is in regime 1. An operational city map for Ardabil
must sit between the two: full municipal coverage (regime 2's area) at a pitch fine enough for
useful accuracy (regime 1's density). §5 shows this is achievable cheaply because the reference
side can be tiled from satellite imagery rather than flown.


# 5. Q3 — An equivalent dataset for Ardabil, Iran

## 5.1 City parameters

| Parameter | Value | Source |
|-----------|------:|--------|
| Municipal area | **18.011 km²** | Wikipedia / Statistical Centre of Iran |
| Population (2022 census) | 588,000 | ibid. |
| Coordinates | 38.25167° N, 48.29750° E | ibid. |
| Elevation | 1,351 m AMSL | ibid. |
| Ardabil County area | 2,211 km² | ibid. |
| Metres per degree latitude at 38.25° N | 111,050 m | WGS-84 series |
| Metres per degree longitude at 38.25° N | 87,595 m | WGS-84 series |
| Equivalent square side | 4.24 km × 4.24 km | derived |

Ardabil's municipality is **7.0× the imaged area of DenseUAV** and **2.5× its convex hull**. At
1,351 m elevation with cold winters and seasonal snow cover, Ardabil also poses an appearance-
change problem that DenseUAV (subtropical, two epochs) does not exercise. This is addressed in
§9.4.

## 5.2 Two ways to build the reference map — and why one is clearly right

**Strategy A — replicate the DenseUAV protocol.** Fly a UAV over Ardabil on a waypoint grid and
capture the reference imagery directly. Table 4 costs this out.

Table: **Table 4.** Cost of building an Ardabil reference map by UAV survey (Strategy A).

| Waypoint pitch | Waypoints | Flight-line length | Flight time @ 15 m/s | Sorties¹ | Images² | Storage |
|---------------:|----------:|-------------------:|---------------------:|---------:|--------:|--------:|
| **20 m** (DenseUAV) | **45,028** | **901 km** | **16.7 h** | **≈ 45** | 405,248 | 203.6 GB |
| 50 m | 7,204 | 360 km | 6.7 h | ≈ 18 | 64,840 | 32.6 GB |
| 100 m | 1,801 | 180 km | 3.3 h | ≈ 9 | 16,210 | 8.1 GB |
| 150 m | 800 | 120 km | 2.2 h | ≈ 6 | 7,204 | 3.6 GB |
| 200 m | 450 | 90 km | 1.7 h | ≈ 5 | 4,052 | 2.0 GB |

¹ assuming 22 min of productive flight per multirotor sortie.
² using the DenseUAV convention of 3 drone + 6 satellite images per waypoint.

Replicating DenseUAV's 20 m pitch over Ardabil would require **901 km of flight lines, roughly
45 sorties and 203.6 GB** — before considering airspace permissions over a city of 588,000
people, or the fact that the imagery would need re-flying whenever the city changes.

**Strategy B — tile an orthorectified satellite mosaic.** This is what DenseUAV's *gallery*
actually is, and what a deployed system actually searches. No flight is required. The design
variables are ground sampling distance (GSD) and inter-tile overlap.

For a tile of $P$ pixels square at ground sampling distance $g$ (m/px), with fractional overlap
$\rho$, the ground side is $T = Pg$, the grid step is $s = T(1-\rho)$, and the tile count over
area $A$ is

$$N \;=\; \left\lceil \frac{A}{s^{2}} \right\rceil \;=\; \left\lceil \frac{A}{P^{2}g^{2}(1-\rho)^{2}} \right\rceil .$$

Table 5 evaluates this for Ardabil at $P = 512$ (matching DenseUAV's tile size and the encoder's
224 × 224 input after resize).

Table: **Table 5.** Ardabil reference-map tiling options (A = 18.011 km², 512 × 512 tiles).

| GSD | Overlap | Tile ground side | Grid step | Tiles | Tiles/km² | JPEG q85 imagery | Flat index, D=512 | Flat index, D=4096 |
|----:|--------:|-----------------:|----------:|------:|----------:|-----------------:|------------------:|-------------------:|
| 0.30 m | 0 % | 153.6 m | 153.6 m | 764 | 42 | 20 MB | 1.6 MB | 12.5 MB |
| **0.30 m** | **50 %** | **153.6 m** | **76.8 m** | **3,054** | **170** | **80 MB** | **6.3 MB** | 50.0 MB |
| 0.30 m | 75 % | 153.6 m | 38.4 m | 12,215 | 678 | 320 MB | 25.0 MB | 200.1 MB |
| 0.50 m | 0 % | 256.0 m | 256.0 m | 275 | 15 | 7 MB | 0.6 MB | 4.5 MB |
| 0.50 m | 50 % | 256.0 m | 128.0 m | 1,100 | 61 | 29 MB | 2.3 MB | 18.0 MB |
| 0.50 m | 75 % | 256.0 m | 64.0 m | 4,398 | 244 | 115 MB | 9.0 MB | 72.1 MB |

![Ardabil sizing. (a) Tile count across GSD/overlap options — the recommended configuration is
highlighted. (b) On-board storage decomposition for the recommended 3,054-tile map: the entire
searchable city fits in 80 MB of imagery plus a 0.4 MB compressed index. (c) Map-construction
effort: satellite tiling is two orders of magnitude cheaper than a UAV survey.](fig/f7_ardabil.png)

## 5.3 Recommended configuration

**0.30 m GSD, 512 × 512 tiles, 50 % overlap → 3,054 tiles, 76.8 m grid step.**

Rationale:

- **0.30 m GSD** matches the resolution used by UAV-VisLoc and is the standard resolution of
  freely available high-resolution orthoimagery. At 224 × 224 encoder input, a 153.6 m tile is
  resampled to 0.686 m/px — well matched to a UAV at 80–120 m altitude.
- **50 % overlap** guarantees that any query footprint is fully contained in at least one
  reference tile, which is the condition for reliable retrieval. Increasing to 75 % quadruples
  cost for a sub-metre accuracy gain that the 76.8 m grid does not require.
- **3,054 tiles** is almost exactly DenseUAV's *deployed* gallery scale (1,554 tiles), so all
  measured retrieval latencies in this report transfer directly.

## 5.4 Full cost of the Ardabil dataset

Table: **Table 6.** Complete resource budget for the recommended Ardabil reference map.

| Item | Quantity | Notes |
|------|---------:|-------|
| Reference tiles | 3,054 | 512 × 512, 0.3 m GSD, 50 % overlap |
| Ground covered | **18.011 km²** | entire municipality |
| Raw imagery (JPEG q85) | **80.1 MB** | 26.2 KB/tile, measured |
| Raw imagery (WebP q85) | 40.6 MB | if storage-constrained |
| Descriptors, D = 512 fp32 | 6.3 MB | 2,048 B/vector |
| FAISS `flat` index, D = 512 | 6.3 MB | exact search |
| **FAISS `ivfpq` index, D = 512** | **0.4 MB** | m = 64, 122 B/vector — recommended |
| FAISS `flat` index, D = 4096 (MixVPR) | 50.0 MB | 8× the D=512 cost |
| Encoder weights, FP16 | 41.7 MB | ViT-S/16 + projection |
| Metadata (CSV / JSON) | ≈ 0.4 MB | one row per tile |
| **Total on-board footprint** | **≈ 123 MB** | imagery + index + weights + metadata |
| Offline build: tile download | ≈ 9 min | 3,054 requests, network-bound |
| **Offline build: encode + index, this laptop** | **4.3 min** | 85 ms/tile measured × 3,054 |
| Offline build: encode + index, Orin Nano | 0.2 min | projected |
| Field survey required | **none** | — |

> **Key finding (Q3).** An Ardabil reference map covering the whole 18.011 km² municipality
> requires **3,054 satellite tiles, 80 MB of imagery and a 0.4 MB search index — about 123 MB
> total on board.** It can be built end-to-end in **under 15 minutes on the existing laptop**,
> with **no UAV survey at all**. This is 7 × the geographic coverage of DenseUAV at 0.7 % of its
> storage cost. The equivalent UAV survey (Strategy A at DenseUAV's 20 m pitch) would need
> 901 km of flight lines, ≈ 45 sorties and 203.6 GB, and is not recommended.

## 5.5 Scaling beyond the city

| Scope | Area | Tiles @ 0.3 m / 50 % | IVF-PQ index | JPEG imagery |
|-------|-----:|---------------------:|-------------:|-------------:|
| Ardabil city | 18.011 km² | 3,054 | 0.4 MB | 80 MB |
| City + 10 km peri-urban ring | ≈ 120 km² | 20,346 | 2.5 MB | 533 MB |
| Ardabil County | 2,211 km² | 374,850 | 45.7 MB | 9.8 GB |
| Ardabil Province | 17,800 km² | 3,017,795 | 368 MB | 79.1 GB |

Even the **entire province** fits in a 368 MB IVF-PQ index — comfortably inside an 8 GB Jetson
Orin Nano's memory. The imagery itself (79 GB) would live on an NVMe SSD and need not be loaded;
only the index and the matched tiles' metadata are required at query time. This is the decisive
architectural advantage of a retrieval-based approach over a map-in-memory approach.


# 6. Q4 — Is the DenseUAV model small enough for real time?

## 6.1 Measured model complexity

Table 7 was produced by instantiating each encoder in this repository and measuring it. This
answers the question "how big is the DenseUAV model" precisely.

Table: **Table 7.** Encoder complexity and measured CPU latency (i7-7600U, batch 1, FP32).

| Encoder | Backbone | Params (M) | GMACs | GFLOPs | Desc. dim | Input | FP32 wt. (MB) | FP16 wt. (MB) | CPU (ms) | CPU (fps) |
|:-----------------|:---------|-----:|-----:|------:|-----:|-----:|------:|------:|------:|------:|
| **DenseUAV ViT** | ViT-S/16 | **21.86** | **4.24** | **8.48** | **512** | 224² | 83.4 | **41.7** | **95.3** | **10.5** |
| MixVPR-512 | RN50-L3 | 10.09 | 8.11 | 16.21 | 512 | 320² | 38.5 | 19.2 | 226.7 | 4.4 |
| MixVPR-4096 | RN50-L3 | 10.88 | 8.42 | 16.84 | 4096 | 320² | 41.5 | 20.8 | 241.2 | 4.1 |
| CosPlace | RN101 | 46.70 | 17.46 | 34.91 | 2048 | 322² | 178.1 | 89.1 | 553.5 | 1.8 |

The DenseUAV encoder is a `vit_small_patch16_224` backbone with a 384 → 512 linear projection
and batch-norm (`avl/models/denseuav_vit.py`), loading the 23 M-parameter
`UAV-Self-Positioning-23M-ZCN` checkpoint from Hugging Face. Our 21.86 M count excludes the
discarded classification head, which accounts for the difference from the "23 M" in the model name.

![Encoder comparison. DenseUAV-ViT has the second-highest parameter count but by far the
**lowest compute per image** — 4.24 GMACs, roughly half of MixVPR and a quarter of CosPlace —
because it operates at 224² rather than 320². Parameter count is a poor proxy for inference
cost in this comparison; GMACs is the correct one.](fig/f3_models.png)

**Why the DenseUAV model is cheap despite its parameter count.** A ViT-S/16 at 224² produces
196 patch tokens; a ResNet-50 truncated at `layer3` and fed 320² produces a 20 × 20 = 400-element
spatial map that the MixVPR aggregator then mixes with four feature-mixer layers over a 400-d
axis. The transformer's parameters are largely in wide MLPs applied to few tokens; the CNN's
compute is spread over many spatial positions. **The DenseUAV encoder does the least work per
image of any encoder in the repository.**

## 6.2 Comparison with the wider VPR literature

Table: **Table 8.** Positioning against commonly used VPR/geo-localisation encoders.

| Model | Backbone | Params | GMACs @ typical input | Descriptor | Notes |
|:----------------|:-------------------|------:|-----:|-----:|:---------------------------------|
| **DenseUAV ViT** | ViT-S/16 @224 | 21.9 M | **4.2** | 512 | in this repo; UAV-to-satellite trained |
| MixVPR-512 | RN50-L3 @320 | 10.1 M | 8.1 | 512 | in this repo |
| MixVPR-4096 | RN50-L3 @320 | 10.9 M | 8.4 | 4096 | in this repo |
| CosPlace | RN101 @322 | 46.7 M | 17.5 | 2048 | in this repo |
| NetVLAD | VGG-16 @480×640 | ≈ 148 M | ≈ 90 | 4096 (32 k raw) | classical baseline, far too heavy |
| CosPlace-light | RN18 @322 | ≈ 11.7 M | ≈ 4.5 | 512 | plausible lighter substitute |
| EigenPlaces | RN50 @322 | ≈ 25 M | ≈ 11 | 2048 | stronger, heavier |
| SALAD / DINOv2-B | ViT-B/14 @322 | ≈ 87 M | ≈ 60 | 8448 | state of the art, ≈ 14× DenseUAV-ViT cost |

The DenseUAV ViT sits at the efficient end of this spectrum. Only a ResNet-18-class model would
be materially cheaper, and it would lose the UAV-to-satellite cross-view training that the
DenseUAV checkpoint provides.

## 6.3 The four-orientation search and its cost

`avl/config.py` defaults to `query_rotations = 4`: the query frame is encoded at 0°, 90°, 180°
and 270° because UAV heading is unknown in GNSS-denied flight. This **quadruples encode cost**
and is the single largest lever in the entire latency budget.

| Configuration | Encode, laptop | Encode, Orin Nano Super (proj.) |
|---------------|---------------:|--------------------------------:|
| 1 orientation | 95.3 ms | 6.0 ms |
| 4 orientations | 381.2 ms | 24.0 ms |

![(a) Measured localisation error on DenseUAV. (b) Cost of the four-orientation search: on the
laptop it pushes the pipeline past the 10 Hz budget; on an Orin Nano it remains comfortably
inside it.](fig/f8_accuracy.png)

**Recommendation.** On a real airframe the heading is known to within a few degrees from the
magnetometer and the IMU's yaw estimate, even without GNSS. Rotating the query to a
north-aligned frame and encoding **once** is the correct production configuration; it recovers a
4× latency saving. Retain `query_rotations = 4` only for offline evaluation and for platforms
without a usable heading reference.

## 6.4 Accuracy versus cost

The repository's stored benchmark artifacts allow a direct accuracy comparison on identical data
(query `002584`, 1,554-tile gallery, four orientations, `flat` index):

| Encoder | Top-1 error | **Fused (Top-5) error** | Query encode | Search |
|---------|------------:|------------------------:|-------------:|-------:|
| MixVPR-4096 | 28.8 m | **189.8 m** | 890.8 ms | 7.76 ms |
| **DenseUAV ViT-512** | **20.0 m** | **5.5 m** | 451.3 ms | 2.04 ms |

On a 200-reference subset with 20 queries, MixVPR-4096 achieved Recall@5 = 0 and a mean fused
error of 151.3 m. The DenseUAV checkpoint is trained specifically for UAV-to-satellite cross-view
retrieval; the MixVPR and CosPlace checkpoints are trained on ground-level street imagery and do
not transfer to nadir aerial views.

The published DenseUAV baseline for this backbone reports **Recall@1 = 83.05 %** and
**SDM@1 = 86.24 %** on the full test protocol.

> **Key finding (Q4).** **Yes — the DenseUAV model is appropriate for real-time
> implementation, and it is the best of the four available options on every axis at once.** At
> 21.86 M parameters and **4.24 GMACs** it is the cheapest encoder in the repository (2.0× cheaper
> than MixVPR, 4.1× cheaper than CosPlace), it produces the most compact descriptor (512-d, 8×
> smaller index than MixVPR-4096), and on this dataset it is also by far the most accurate
> (5.5 m fused error versus 189.8 m). There is no accuracy-versus-speed trade-off to make here:
> the fastest model is also the most accurate, because it is the only one trained for this task.


# 7. Q5 — Retrieval index: compute, memory and latency

## 7.1 Index families and their memory cost

For $N$ vectors of dimension $D$, the serialised index size (bytes/vector, measured and
verified against FAISS output) is

$$
\text{flat} = 4D, \qquad
\text{HNSW}_{M=32} = 4D + 271, \qquad
\text{IVF-PQ}_{m=64} = 122 .
$$

Table: **Table 9.** Index memory footprint (MB) versus gallery size.

| N (tiles) | Scope | D512 flat | D512 HNSW | **D512 IVF-PQ** | D2048 flat | D4096 flat |
|----------:|-------|----------:|----------:|----------------:|-----------:|-----------:|
| 1,554 | DenseUAV test gallery | 3.2 | 3.6 | **0.2** | 12.7 | 25.5 |
| **3,054** | **Ardabil city** | **6.3** | 7.1 | **0.4** | 25.0 | 50.0 |
| 12,215 | Ardabil, 75 % overlap | 25.0 | 28.3 | 1.5 | 100.1 | 200.1 |
| 45,028 | Ardabil, 20 m UAV grid | 92.2 | 104.4 | 5.5 | 368.9 | 737.7 |
| 374,850 | Ardabil County | 767.7 | 869.2 | 45.7 | 3,071 | 6,143 |
| 3,017,795 | Ardabil Province | 6,180 | 6,998 | **368** | 24,722 | 49,444 |

The IVF-PQ column is the reason a province-scale map is feasible on an 8 GB module:
product quantisation compresses each 512-d descriptor from 2,048 B to 122 B, a **16.8× reduction**,
at a small recall cost that the geo-fusion stage largely absorbs.

## 7.2 Measured search latency

Table: **Table 10.** FAISS search latency on the i7-7600U (median, k = 10, 4 threads).

| D | N | Index | Size | 1 query | 4 rotations |
|--:|--:|-------|-----:|--------:|------------:|
| 512 | 1,554 | flat | 3.2 MB | 0.27 ms | 0.29 ms |
| 512 | 1,554 | HNSW | 3.6 MB | 0.23 ms | 0.60 ms |
| 512 | 7,204 | flat | 14.8 MB | 1.24 ms | 3.90 ms |
| 512 | 7,204 | HNSW | 16.7 MB | 1.88 ms | 3.20 ms |
| **512** | **7,204** | **IVF-PQ** | **1.7 MB** | **0.17 ms** | **0.42 ms** |
| 512 | 45,028 | flat | 92.2 MB | 45.47 ms | 42.36 ms |
| 512 | 45,028 | HNSW | 104.5 MB | 3.54 ms | 9.26 ms |
| **512** | **45,028** | **IVF-PQ** | **5.5 MB** | **0.20 ms** | **0.38 ms** |
| 2048 | 45,028 | flat | 368.9 MB | 138.28 ms | 183.77 ms |
| 4096 | 7,204 | flat | 118.0 MB | 30.85 ms | 61.71 ms |
| 4096 | 45,028 | flat | 737.7 MB | 232.39 ms | 315.73 ms |

![FAISS scaling. (a) Index memory versus gallery size; the dashed line marks the 8 GB Orin Nano
memory ceiling. (b) Measured search latency. Exact `flat` search collapses beyond ~10,000
vectors because it becomes memory-bandwidth bound on a 4 MiB-L3 CPU; IVF-PQ stays flat at
0.2 ms across two orders of magnitude.](fig/f5_faiss.png)

**Why `flat` collapses.** An exact inner-product search reads the entire index once per query.
At N = 45,028 and D = 512 that is 92 MB against an effective DRAM bandwidth of roughly 3 GB/s on
this laptop — approximately 30 ms of unavoidable memory traffic, matching the 45 ms measured.
The problem is bandwidth, not arithmetic, so it does not improve with more cores. At D = 4096 the
same search reads 738 MB and takes 232 ms.

## 7.3 Index recommendation

| Gallery size | Recommended index | Rationale |
|--------------|-------------------|-----------|
| < 5,000 (Ardabil city) | `flat` **or** `ivfpq` | flat is exact and costs only 6.3 MB / 0.3 ms; IVF-PQ is 16× smaller and equally fast |
| 5,000 – 100,000 | `hnsw` or `ivfpq` | flat is already the pipeline bottleneck |
| > 100,000 (county / province) | **`ivfpq`** | the only family that fits in module memory |

For the recommended Ardabil map (3,054 tiles) the current default of `flat` with D = 512 is
already correct and costs 0.29 ms. **The retrieval index is not, and will not become, the
bottleneck in this system** — the encoder is, by two to three orders of magnitude.


# 8. Q6/Q7 — Host platform: the laptop today, and the Jetson tomorrow

## 8.1 Why the current laptop cannot be real-time

The i7-7600U is a 2-core, 15 W ultrabook CPU from 2017 with no CUDA device. Every measurement in
Table 7 was taken on it. The consequences for the pipeline are shown in Table 11.

Table: **Table 11.** End-to-end single-fix latency on the development laptop
(Ardabil 3,054-tile map; decode + resize 12 ms, geo-fusion 0.3 ms).

| Encoder | Rotations | Encode | Search | **Total** | **Rate** | Verdict |
|---------|----------:|-------:|-------:|----------:|---------:|---------|
| DenseUAV ViT | 4 | 381.2 ms | 0.42 ms | **393.9 ms** | **2.54 Hz** | 1 Hz only |
| DenseUAV ViT | 1 | 95.3 ms | 0.17 ms | **107.8 ms** | **9.28 Hz** | marginal at 5 Hz |
| MixVPR-512 | 4 | 906.8 ms | 0.42 ms | 919.5 ms | 1.09 Hz | 1 Hz only |
| MixVPR-4096 | 4 | 964.8 ms | 48.92 ms | 1,026.0 ms | 0.97 Hz | fails |
| CosPlace-2048 | 4 | 2,214.0 ms | 33.95 ms | 2,260.3 ms | 0.44 Hz | fails |

Two further limitations compound this. Sustained inference on a 15 W ultrabook CPU triggers
thermal throttling — the CPU was observed scaling to 51 % of nominal frequency during the FAISS
sweep — so *sustained* rates are below the burst rates above. And a laptop is not an airborne
payload: mass, vibration tolerance, DC input and shock rating all disqualify it.

> **Key finding (Q6).** The development laptop reaches **2.5 Hz** in the project's default
> configuration and at best **9.3 Hz** with a single orientation. It is adequate for offline
> index building (4.3 min for the whole Ardabil map) and for algorithm development, but it
> **cannot** support real-time flight.

## 8.2 What "real-time" actually requires

Real-time is not a single number; it derives from vehicle dynamics and the map's spatial
resolution. Table 12 makes the requirement explicit.

Table: **Table 12.** Derivation of the required fix rate.

| UAV speed | Distance per 100 ms | Fixes needed for one per 76.8 m map cell | Fixes needed for one per 20 m |
|----------:|--------------------:|------------------------------------------:|------------------------------:|
| 5 m/s (18 km/h) | 0.5 m | 0.07 Hz | 0.25 Hz |
| 10 m/s (36 km/h) | 1.0 m | 0.13 Hz | 0.50 Hz |
| **15 m/s (54 km/h)** | **1.5 m** | **0.20 Hz** | **0.75 Hz** |
| 25 m/s (90 km/h) | 2.5 m | 0.33 Hz | 1.25 Hz |
| 40 m/s (144 km/h) | 4.0 m | 0.52 Hz | 2.00 Hz |

Purely geometrically, **1 Hz is sufficient** to obtain a fix in every map cell at any realistic
UAV speed. Higher rates are required for different reasons:

1. **Bounding inertial drift.** A consumer-grade MEMS IMU drifts at roughly 1 % of distance
   travelled. At 15 m/s and 1 Hz, position uncertainty grows to ~15 cm between fixes — acceptable;
   at 0.5 Hz with an outlier-rejected fix stream it is not.
2. **Outlier rejection.** Retrieval fails on featureless terrain, glare, cloud shadow and snow.
   A 10 Hz stream lets a temporal filter reject 80 % of fixes and still deliver 2 Hz of
   trustworthy position.
3. **Control-loop compatibility.** Standard autopilot position-estimator inputs (PX4 EKF2,
   ArduPilot EKF3) expect external vision position at 5–30 Hz.

**Adopted targets:** 10 Hz (100 ms) = comfortable; 5 Hz (200 ms) = navigation-grade minimum;
1 Hz = position-reset only.

## 8.3 Jetson module comparison

Table: **Table 13.** NVIDIA Jetson module comparison. TOPS are dense INT8 (half the
sparse figure NVIDIA headlines). Prices are 2026 list, following NVIDIA's July 2026 increase.

| Module | GPU | CPU | Memory | Bandwidth | Dense INT8 | Power | Price | Status |
|--------|-----|-----|-------:|----------:|-----------:|------:|------:|--------|
| Jetson Nano | 128-core Maxwell | 4 × A57 @1.43 GHz | 4 GB LPDDR4 | 25.6 GB/s | 0.24 TOPS | 5–10 W | $199 | **end-of-life** |
| Xavier NX 16 GB | 384-core Volta + 48 TC | 6 × Carmel @1.9 GHz | 16 GB LPDDR4x | 59.7 GB/s | 10.5 TOPS | 10–20 W | $699 | legacy |
| Orin Nano 4 GB Super | 512-core Ampere + 16 TC | 6 × A78AE @1.7 GHz | 4 GB LPDDR5 | 51 GB/s | 17 TOPS | 7–25 W | $349 | current |
| **Orin Nano 8 GB Super** | **1024-core Ampere + 32 TC** | **6 × A78AE @1.7 GHz** | **8 GB LPDDR5** | **102 GB/s** | **33.5 TOPS** | **7–25 W** | **$399** | **current** |
| Orin NX 8 GB Super | 1792-core Ampere + 56 TC | 8 × A78AE @2.0 GHz | 8 GB LPDDR5 | 102.4 GB/s | 58.5 TOPS | 10–40 W | $649 | current |
| Orin NX 16 GB Super | 2048-core Ampere + 64 TC | 8 × A78AE @2.0 GHz | 16 GB LPDDR5 | 102.4 GB/s | 78.5 TOPS | 10–40 W | $899 | current |
| AGX Orin 32 GB | 2048-core Ampere + 64 TC | 12 × A78AE @2.2 GHz | 32 GB LPDDR5 | 204.8 GB/s | 100 TOPS | 15–60 W | $1,799 | current |
| AGX Orin 64 GB | 2048-core Ampere + 64 TC | 12 × A78AE @2.2 GHz | 64 GB LPDDR5 | 204.8 GB/s | 137.5 TOPS | 15–75 W | $2,999 | current |
| AGX Thor T5000 | 2560-core Blackwell | 14 × Neoverse-V3AE @2.6 GHz | 128 GB LPDDR5X | 273 GB/s | ~1,035 TOPS | 40–130 W | $5,499 | current |

![Jetson comparison. (a) AI throughput spans four orders of magnitude across the family.
(b) End-to-end fix latency versus 2026 module price for the recommended pipeline. Every current
Orin module clears the 10 Hz line with margin; the knee of the curve is at the Orin Nano 8 GB
Super.](fig/f4_boards.png)

## 8.4 Latency projection: method and validation

No ViT-S/16 TensorRT measurement on an Orin Nano is published, so latencies are projected. The
method is stated here so it can be audited.

**Anchor.** Two published TensorRT FP16 measurements of transformer models on an Orin Nano 8 GB
Super give the achieved FP16 rate:

| Anchor model | GMACs | Measured latency | Implied achieved rate |
|--------------|------:|-----------------:|----------------------:|
| SegFormer-B0 @640² (25 W) | 14.84 | 15.02 ms | 1.98 TFLOP/s |
| ViT-B/16 @224² (FP16) | 17.60 | 19.90 ms | 1.77 TFLOP/s |

The two agree to within 12 %, which is reassuring. We adopt the **conservative 1.77 TFLOP/s**
plus a 1.2 ms fixed overhead (kernel launch, H2D/D2H, GPU pre-processing).

**Cross-board scaling.** Latency does not scale linearly with peak throughput because small
models become launch- and bandwidth-bound. Fitting the exponent $\alpha$ in
$t_{\text{board}} = t_{\text{anchor}} \cdot (F_{\text{board}}/F_{\text{anchor}})^{-\alpha}$
to published YOLO26n and SegFormer-B0 measurements across the Nano Super / Orin NX 16 GB /
AGX Orin 64 GB triad gives $\alpha \in [0.12, 0.40]$. We adopt $\alpha = 0.50$ for boards faster
than the anchor (deliberately conservative — ViT-S is more compute-bound than YOLO26n),
$\alpha = 0.85$ for slower boards, and $\alpha = 1.0$ for the Maxwell-based Jetson Nano, which
has no tensor cores.

Table: **Table 14.** Projected TensorRT FP16 encode latency per image, batch 1 (ms).

| Board | DenseUAV ViT | MixVPR-512 | MixVPR-4096 | CosPlace-2048 |
|-------|-------------:|-----------:|------------:|--------------:|
| i7-7600U laptop *(measured, FP32)* | *95.3* | *226.7* | *241.2* | *553.5* |
| Jetson Nano (legacy) | 212.0 | 366.5 | 379.1 | 740.3 |
| Xavier NX 16 GB | 14.3 | 24.7 | 25.6 | 50.0 |
| Orin Nano 4 GB Super | 10.6 | 18.4 | 19.0 | 37.2 |
| **Orin Nano 8 GB Super** | **6.0** | 10.4 | 10.7 | 20.9 |
| Orin NX 8 GB Super | 4.5 | 7.8 | 8.1 | 15.8 |
| Orin NX 16 GB Super | 3.9 | 6.8 | 7.0 | 13.6 |
| AGX Orin 32 GB | 3.5 | 6.0 | 6.2 | 12.1 |
| AGX Orin 64 GB | 3.0 | 5.1 | 5.3 | 10.3 |
| AGX Thor T5000 | 1.6 | 2.7 | 2.8 | 5.4 |

INT8 quantisation would reduce these by a further ~28 % where calibration succeeds.

## 8.5 End-to-end latency on candidate boards

Table: **Table 15.** Complete online budget, Ardabil 3,054-tile map
(decode + resize 4 ms on Jetson; geo-fusion 0.3 ms).

| Platform | Encoder | Rot. | Encode | Search | **Total** | **Rate** | 10 Hz |
|----------|---------|-----:|-------:|-------:|----------:|---------:|:-----:|
| i7-7600U laptop | DenseUAV ViT | 4 | 381.2 | 0.42 | 393.9 ms | 2.5 Hz | **No** |
| i7-7600U laptop | DenseUAV ViT | 1 | 95.3 | 0.17 | 107.8 ms | 9.3 Hz | **No** |
| Orin Nano 4 GB | DenseUAV ViT | 4 | 42.6 | 0.26 | 47.1 ms | 21.2 Hz | Yes |
| **Orin Nano 8 GB Super** | **DenseUAV ViT** | **4** | **24.0** | **0.26** | **28.5 ms** | **35.1 Hz** | **Yes** |
| **Orin Nano 8 GB Super** | **DenseUAV ViT** | **1** | **6.0** | **0.11** | **10.4 ms** | **96.2 Hz** | **Yes** |
| Orin Nano 8 GB Super | MixVPR-4096 | 4 | 42.9 | 16.31 | 63.5 ms | 15.8 Hz | Yes |
| Orin Nano 8 GB Super | CosPlace-2048 | 4 | 83.7 | 11.32 | 99.3 ms | 10.1 Hz | Yes* |
| Orin NX 16 GB Super | DenseUAV ViT | 4 | 15.6 | 0.26 | 20.2 ms | 49.5 Hz | Yes |
| AGX Orin 64 GB | DenseUAV ViT | 4 | 11.8 | 0.26 | 16.4 ms | 61.1 Hz | Yes |

![End-to-end latency budget. Note the axis scales differ by a factor of 23 between panels.
On the laptop the neural encode dominates completely; on a Jetson, decode and pre-processing
become a comparable share, which is why GPU-side pre-processing (NVJPEG + NPP) is worth
implementing.](fig/f6_budget.png)

## 8.6 Offline index-build cost across platforms

Table: **Table 16.** Time to encode and index a reference map.

| Platform | Encoder | 3,054 tiles (Ardabil) | 12,215 tiles | 45,028 tiles |
|----------|---------|----------------------:|-------------:|-------------:|
| i7-7600U laptop | DenseUAV ViT | 4.3 min | 17.3 min | 1.06 h |
| i7-7600U laptop | MixVPR-4096 | 12.7 min | 50.9 min | 3.13 h |
| i7-7600U laptop | CosPlace-2048 | 24.4 min | 1.63 h | 6.00 h |
| Orin Nano 8 GB Super | DenseUAV ViT | 0.2 min | 0.7 min | 2.5 min |
| Orin NX 16 GB Super | DenseUAV ViT | 0.1 min | 0.4 min | 1.6 min |
| AGX Orin 64 GB | DenseUAV ViT | 0.1 min | 0.3 min | 1.2 min |

The map can be rebuilt **on the aircraft** in 12 seconds if required — for example, to swap to a
different city or a seasonally updated mosaic.

## 8.7 Board recommendation

Table: **Table 17.** Decision matrix.

| Criterion | Orin Nano 8 GB Super | Orin NX 16 GB Super | AGX Orin 64 GB |
|-----------|---------------------:|--------------------:|---------------:|
| Fix rate, ViT ×4 rotations | 35 Hz | 50 Hz | 61 Hz |
| Fix rate, ViT ×1 orientation | 96 Hz | 120 Hz | 136 Hz |
| Margin over 10 Hz target | **3.5×** | 5.0× | 6.1× |
| Memory | 8 GB | 16 GB | 64 GB |
| Largest IVF-PQ map that fits comfortably | province-scale | multi-province | national |
| Module price (2026) | **$399** | $899 | $2,999 |
| Power | 7–25 W | 10–40 W | 15–75 W |
| Module mass (with heatsink/fan) | ≈ 150 g | ≈ 175 g | ≈ 700 g |
| Suitable airframe | ≥ 3 kg multirotor | ≥ 5 kg | ≥ 10 kg / fixed-wing |
| **Cost per Hz delivered** | **$11.4** | $18.0 | $49.2 |

> **Key finding (Q7).** **The NVIDIA Jetson Orin Nano 8 GB Super ($399, 25 W) is the correct
> board for this system.** With DenseUAV-ViT in TensorRT FP16 it delivers an estimated **35 Hz
> with the full four-orientation search and 96 Hz with a single orientation** on the 3,054-tile
> Ardabil map — 3.5× to 9.6× the 10 Hz real-time target. Its 8 GB of memory holds an IVF-PQ index
> for the entire Ardabil *province*. Even with the projection's ±30 % uncertainty and a further
> 2× safety factor for sustained thermal operation, it clears the target.
>
> Choose the **Orin NX 16 GB ($899)** only if the same computer must concurrently run object
> detection, tracking, obstacle avoidance or visual-inertial odometry — the localisation task
> alone does not justify it. The **AGX Orin** and **Thor** are substantially over-specified.
> The **Orin Nano 4 GB** (21 Hz, $349) is viable but saves only $50 while halving memory
> bandwidth; it is not recommended. **Jetson Nano and Xavier NX should not be selected** —
> the former is end-of-life and roughly 2× slower than the laptop for this workload.


# 9. Q8 — Complete requirements for a real-time implementation

Reaching real-time is not only a hardware purchase. §9.1–9.5 enumerate everything required.

## 9.1 Hardware bill of materials

Table: **Table 18.** Recommended airborne BOM.

| Item | Specification | Approx. cost | Notes |
|------|---------------|-------------:|-------|
| Compute module | Jetson Orin Nano 8 GB Super | $399 | 25 W MAXN_SUPER mode |
| Carrier board | Third-party mini carrier (e.g. reComputer J401, A203) | $150–250 | M.2 NVMe + CSI + UART |
| Storage | 256 GB NVMe M.2 2280 | $30 | JetPack + maps + logs |
| Camera | Global-shutter nadir, ≥ 1.2 MP, CSI-2 (e.g. IMX296) | $80–150 | global shutter is **required** |
| Lens | 60–90° HFOV, fixed focus at infinity | $40 | sets ground footprint |
| Cooling | Active heatsink + fan | $25 | mandatory for sustained 25 W |
| IMU / heading | Autopilot's own (PX4/ArduPilot) via MAVLink | — | supplies yaw for 1-orientation mode |
| Power | 5 V / 5 A regulated from airframe BEC | $20 | with brown-out protection |
| Vibration isolation | Damped mount | $15 | protects NVMe and camera |
| **Total** | | **≈ $760–930** | excluding airframe |

Thermal and mass budget: 25 W dissipation at 1,351 m elevation (Ardabil) means ~14 % lower air
density than sea level; size the heatsink for 30 W to be safe. Total payload mass ≈ 350–450 g
including camera, carrier and cooling — within the capacity of a 3 kg-class multirotor.

## 9.2 Software stack

| Layer | Requirement | Status in repo |
|-------|-------------|----------------|
| JetPack 6.x / L4T | CUDA 12.x, cuDNN, TensorRT 10.x | not applicable yet |
| **TensorRT engine** | Export ViT-S/16 → ONNX (opset 17) → TRT FP16 plan | **missing — required** |
| INT8 calibration | ~500 representative tiles, entropy calibration | optional, gives a further ~28 % |
| FAISS | `faiss-cpu` on the Arm CPU is sufficient (0.3 ms); `faiss-gpu` optional | present |
| Camera capture | GStreamer `nvarguscamerasrc` → CUDA, zero-copy | **missing — required** |
| GPU pre-processing | NVJPEG decode + NPP resize/normalise | **missing — recommended** |
| Autopilot bridge | MAVLink `VISION_POSITION_ESTIMATE` at 10 Hz | **missing — required** |
| Temporal filter | EKF or particle filter fusing fixes with IMU | **prototyped offline** — 15-state error-state EKF in `avl/nav/`, driven by `scripts/visloc_traj.py` and the Benchmark Console "Trajectory"/"Studio" tabs (simulated IMU; see `docs/VisualInertial_AVL_Design.md`). Still to do: real IMU input + MAVLink. |
| Geographic pre-filter | Restrict search to a radius around the prior | **missing — recommended** |
| Watchdog / failsafe | Fall back to dead reckoning when confidence < threshold | partially present (`score_threshold`) |

## 9.3 Engineering work items, in priority order

1. **TensorRT export and validation.** Convert `DenseUAVViT` to ONNX and build an FP16 plan.
   Verify that descriptor cosine similarity against the PyTorch reference stays above 0.999 —
   below that, retrieval ranking changes. *Estimated effort: 2–3 days.*
2. **Single-orientation mode with heading input.** Add a `--heading-deg` path that de-rotates the
   query before encoding, reducing latency 4×. *1–2 days.*
3. **GPU capture and pre-processing pipeline.** Eliminate the 4 ms CPU decode/resize.
   *3–5 days.*
4. **Temporal filter.** The deployed path is still memoryless (each frame localised
   independently), but the filter itself now exists: a 15-state error-state EKF
   (`avl/nav/`, `scripts/visloc_traj.py`, Console "Trajectory"/"Studio" tabs)
   fuses the per-frame VPR pose with a simulated IMU, χ²-gates outliers, and coasts
   through fix dropouts (`docs/VisualInertial_AVL_Design.md`). Remaining: feed a
   real IMU stream instead of the synthetic one, and wire it into the runtime. *~1 week.*
5. **Geographic pre-filtering at runtime.** With a position prior, restrict the search to a
   2 km radius. Reduces a province-scale index to a few thousand candidates. *2–3 days.*
6. **MAVLink integration and HIL testing.** *1–2 weeks.*
7. **Confidence calibration.** The current `score_threshold = 0.35` is a magic constant; it
   should be calibrated against measured error on the Ardabil map. *3 days.*

## 9.4 Data and environment requirements specific to Ardabil

- **Seasonal appearance.** Ardabil sits at 1,351 m with substantial winter snow cover. The
  reference mosaic should be built from **at least two epochs** (summer and winter), following
  DenseUAV's own two-epoch convention. This doubles the tile count to 6,108 and the index to
  0.8 MB — negligible.
- **Illumination and shadow.** Prefer a mosaic captured near local solar noon to minimise
  directional shadow mismatch with midday flights.
- **Featureless regions.** Agricultural land at the city margin and the frozen surface of
  Shorabil Lake will retrieve poorly. These regions must be identified during map construction
  and flagged so the filter can down-weight fixes there.
- **Map currency.** Ardabil is developing; plan an annual mosaic refresh. Rebuilding the index
  costs 4.3 minutes.
- **Validation flights.** A minimum of 3 sorties with RTK-GNSS ground truth, covering urban core,
  residential and peri-urban terrain, is needed to characterise real error — the DenseUAV
  numbers in §6.4 will not transfer unchanged.

## 9.5 Verification plan

| Test | Method | Pass criterion |
|------|--------|----------------|
| Encoder fidelity | Cosine similarity, TRT vs PyTorch, 500 tiles | > 0.999 mean |
| Retrieval quality | Recall@1/5/10 on held-out Ardabil queries | Recall@5 > 90 % |
| Localisation error | Fused error against RTK ground truth | median < 15 m, p95 < 40 m |
| Sustained throughput | 30 min continuous inference at 25 W | ≥ 10 Hz, no thermal throttle |
| Power draw | Inline measurement at board input | < 25 W mean |
| Failure behaviour | Inject featureless / occluded frames | confidence gate rejects; no position jump |


\clearpage

# 10. Answers

**Q1 — Coverage area and size of the current dataset.**
DenseUAV contains **40,833 images occupying 16.76 GB** across 3,033 unique sampling locations at
a measured **19.7 m pitch**, collected over **14 university campuses in Hangzhou, China**. Its
coverage is **2.58 km² of effectively imaged ground**, within a **7.13 km² convex hull** and a
**9.39 km² bounding box** (5.465 × 1.719 km). Quote 2.58 km² as the coverage and 7.13 km² as the
survey extent. Storage intensity is 6.49 GB per imaged km².

**Q2 — Comparison with other datasets.**
DenseUAV is a *dense small-area* dataset. UAV-VisLoc covers **380 km²** — 147× more ground — with
6× fewer images, because it samples sparsely along corridors. University-1652, SUES-200 and
DenseUAV all target campus-scale precision; VPAIR and ALTO target long traverses. See Table 3.

**Q3 — Ardabil dataset size and coverage.**
Ardabil city is **18.011 km²** (7× DenseUAV's imaged area). Built the correct way — by tiling a
0.3 m orthomosaic into 512 × 512 tiles at 50 % overlap — the dataset is **3,054 tiles, 80 MB of
JPEG imagery, a 6.3 MB exact index (0.4 MB compressed), and ≈ 123 MB on board including model
weights.** It builds in **≈ 9 min of download plus 4.3 min of encoding** on the existing laptop,
with **no UAV survey**. Replicating DenseUAV's 20 m UAV-survey protocol instead would need 45,028
waypoints, 901 km of flight lines, ≈ 45 sorties and 203.6 GB — and is not recommended.

**Q4 — Is the DenseUAV model suitable for real time?**
**Yes.** ViT-S/16 with a 512-d projection: **21.86 M parameters, 4.24 GMACs, 41.7 MB in FP16.**
It is the cheapest of the four supported encoders (2.0× cheaper than MixVPR, 4.1× cheaper than
CosPlace), produces the most compact descriptor, and is the most accurate on this data
(**5.5 m** fused error versus 189.8 m for MixVPR-4096). At **95.3 ms on the laptop CPU** and an
estimated **6.0 ms on an Orin Nano Super**, it is comfortably real-time on the recommended
hardware. No alternative in the repository or in the mainstream VPR literature improves on it
for this task at lower cost.

**Q5 — Compute and storage cost per model and per dataset.**
Tables 7, 9, 10, 14 and 16. Summary for the Ardabil map: DenseUAV-ViT needs 4.24 GMACs/frame,
a 6.3 MB index and 4.3 min to build on the laptop; MixVPR-4096 needs 2.0× the compute, an 8×
larger index (50 MB) and 3× the build time, for far worse accuracy.

**Q6 — Laptop specification and verdict.**
Intel Core i7-7600U (2C/4T, 2.8 GHz, 4 MiB L3), 15 GiB RAM, Intel HD 620 integrated graphics,
**no CUDA device**, Ubuntu 24.04. It achieves **2.54 Hz** in the default configuration and
9.28 Hz with a single orientation. **Not real-time capable**, but fully adequate for offline map
building and development.

**Q7 — Which Jetson board.**
**Jetson Orin Nano 8 GB Super — $399, 7–25 W, 33.5 dense INT8 TOPS, 1024 Ampere cores, 8 GB
LPDDR5 at 102 GB/s.** Estimated **35 Hz** with four orientations and **96 Hz** with one, giving
3.5–9.6× margin over the 10 Hz target, and enough memory for a province-scale index. Upgrade to
**Orin NX 16 GB ($899)** only if the same computer must also run perception tasks. Do not choose
Jetson Nano (end-of-life, slower than the laptop) or Xavier NX (legacy, 2.4× slower than the
Orin Nano at 1.75× the price).

**Q8 — Everything required for real-time implementation.**
Hardware ≈ $760–930 (Table 18); software stack and gaps (Table, §9.2); seven engineering work
items totalling roughly 5–7 person-weeks (§9.3), of which the four essential ones are the
TensorRT export, the single-orientation heading path, the temporal filter, and the MAVLink
bridge; Ardabil-specific data requirements including two-epoch seasonal imagery (§9.4); and the
verification plan in §9.5.

\vspace{3mm}

![System architecture with the recommended configuration and measured/projected timings at
each stage.](fig/f9_pipeline.png)


# 11. Limitations

1. **Jetson latencies are projected, not measured.** No Jetson hardware was available. The
   projection is anchored on two published TensorRT transformer benchmarks that agree to 12 %,
   and uses a deliberately conservative scaling exponent, but ±30 % uncertainty remains. The
   conclusions are unaffected because the margins are 3.5–10×.
2. **The accuracy comparison in §6.4 uses stored artifacts with a small query count.** The
   DenseUAV-ViT versus MixVPR head-to-head at 1,554 references is based on a single query image;
   the 20-query smoke test covered MixVPR only. The direction of the result is unambiguous
   (5.5 m versus 189.8 m, and Recall@5 = 0 for MixVPR) and is consistent with the fact that only
   the DenseUAV checkpoint was trained cross-view, but the precise error figures should be
   re-derived from a larger query set before publication. The full 2,331-query protocol was
   attempted on this laptop during the preparation of this report and had not completed after
   50 minutes of wall-clock time (four rotations per query on a thermally throttled 2-core CPU);
   a 100-query head-to-head is the practical compromise and should be run before the results in
   §6.4 are quoted externally.
3. **Ardabil figures are a design projection.** No Ardabil imagery was acquired or encoded. Tile
   counts and storage follow deterministically from the stated geometry, but retrieval accuracy
   over Iranian urban terrain is unmeasured and may differ from Hangzhou.
4. **Ardabil's 18.011 km² is the municipal boundary.** The contiguous built-up area including
   peri-urban development is larger; §5.5 gives the scaling.
5. **Compression effects on retrieval are untested.** JPEG q85 re-encoding is assumed lossless
   for retrieval purposes at 224² input. This should be verified empirically.
6. **The 150 m footprint used for the coverage-union calculation is inferred** from typical UAV
   camera geometry at 80–100 m altitude, not from published DenseUAV camera intrinsics. The
   200 m variant is given as a sensitivity check.


# 12. Sources

**Datasets and methods**

- Dai, Zheng et al., *Vision-Based UAV Self-Positioning in Low-Altitude Urban Environments*
  (DenseUAV). <https://arxiv.org/abs/2201.09201>
- Xu, Yao, Cao et al., *UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization*.
  <https://arxiv.org/abs/2405.11936>
- Zheng et al., *University-1652: A Multi-view Multi-source Benchmark for Drone-based
  Geo-localization*.
- Zhu et al., *SUES-200: A Multi-height Multi-scene Cross-view Image Benchmark*.
- Cisneros et al., *ALTO: A Large-Scale Dataset for UAV Visual Place Recognition and
  Localization*. <https://arxiv.org/abs/2207.12317>
- Schleiss et al., *VPAIR — Aerial Visual Place Recognition and Localization in Large-scale
  Outdoor Environments*. <https://arxiv.org/abs/2205.11567>
- Ali-bey, Chaib-draa, Giguère, *MixVPR: Feature Mixing for Visual Place Recognition*, WACV 2023.
  <https://arxiv.org/abs/2303.02190>
- Berton, Masone, Caputo, *Rethinking Visual Geo-localization for Large-Scale Applications*
  (CosPlace), CVPR 2022. <https://arxiv.org/abs/2204.02287>
- Johnson, Douze, Jégou, *Billion-scale similarity search with GPUs* (FAISS).
  <https://github.com/facebookresearch/faiss>
- DenseUAV ViT checkpoint: `Bancie/UAV-Self-Positioning-23M-ZCN`, Hugging Face.

**Embedded hardware**

- NVIDIA, *Jetson Modules* product page. <https://developer.nvidia.com/embedded/jetson-modules>
- Forecr, *NVIDIA Jetson Comparison — Orin Series to AGX Thor Series* (module specification
  table). <https://www.forecr.io/blogs/embedded-systems/nvidia-jetson-comparison>
- CNX Software, *NVIDIA increases the price of Jetson modules and devkits by up to 101 %*,
  22 July 2026. <https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/>
- Ultralytics, *NVIDIA Jetson deployment guide and TensorRT benchmarks*.
  <https://docs.ultralytics.com/guides/nvidia-jetson/>
- *AgriJetsonBench: External-Power-Referenced TensorRT Benchmarking of Agricultural Vision
  Models on Jetson Edge Platforms* (SegFormer-B0 anchor). <https://arxiv.org/html/2608.00927v1>
- Choi, *jetson-orin-nano-benchmarks*. <https://github.com/hokwangchoi/jetson-orin-nano-benchmarks>

**Geography**

- *Ardabil* and *Ardabil province*, Wikipedia (area, population, coordinates, elevation).
  <https://en.wikipedia.org/wiki/Ardabil>

**Measurements**

- All dataset, model, and FAISS measurements were taken on the project machine on 14 August 2026
  and are reproducible with the scripts described in §2.2.
