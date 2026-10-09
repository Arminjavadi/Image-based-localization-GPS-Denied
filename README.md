# Image-based Localization (GPS-Denied)

Absolute Visual Localization (AVL) for UAVs using geo-tagged reference imagery, state-of-the-art visual place recognition (VPR), and FAISS approximate nearest-neighbor search.

## Pipeline

```
Geo-tagged reference tiles ──► encoder or ensemble ──► − map mean ──► FAISS index (flat / HNSW / IVF-PQ)

UAV frame + heading + altitude + IMU prior
   │
   ▼
north-up rotation ─► altitude crop ─► encode ─► − flight mean ─► search inside the prior's window
   ─► [re-rank] ─► top-5 cluster fusion ─► lat / lon
```

## The recipe — what the progress report measured

Every step around the encoder is training-free and was measured on UAV-VisLoc (top-1 within
100 m, 144 frames per region; [docs/AVL_Progress_Report](docs/AVL_Progress_Report.html)). The
definitions live in one place, `avl/pipeline.py`, and are used by the benchmark
(`scripts/visloc_eval.py`), the single-image tool (`scripts/visloc_query.py`), the on-board
localizer (`avl.localizer.AVLLocalizer`) and the console, so none of them can drift apart.

| Step | What it does | Measured |
|------|--------------|----------|
| Encoder ensemble | several encoders, cosine scores averaged (`avl/ensemble.py`) | r05 59.7 → 80.6 % (MegaLoc + Game4Loc + AnyLoc-L) |
| Heading alignment | turn the frame north-up from the compass; 1 encode instead of 4 | MegaLoc r05 56.9 → 68.1 % |
| Domain centering | subtract the map's / the flight's mean descriptor (`avl/centering.py`) | r05 68.1 → 74.3 %, r10 16.0 → 27.8 % |
| Altitude scale | crop the frame to the tile's ground size when it covers > 1.25× a tile | r11 26.4 → 65.3 % |
| IMU search window | search only tiles within 3σ of the navigation prior | +12–19 pts at σ = 100 m |
| Top-5 fusion | keep the largest geo-consistent group of the top-5 | ensemble r05 85.4 → 91.7 % |

Three presets (`avl.pipeline.PRESETS`): **baseline** (MegaLoc, 4-rotation search),
**recommended** (MegaLoc + heading + flight centering + gated altitude crop — the shipped
pipeline, mean of four regions 45 → 55 %) and **accuracy** (the three-encoder ensemble with the
same steps, 3.8× the compute). `scripts/check_localizer_parity.py` checks that the localizer
reproduces the benchmark frame by frame.

## Features

- **The recipe** (`avl/pipeline.py`) — encoder ensembles, heading alignment, domain centering,
  altitude scale, IMU search window, cluster fusion; see [The recipe](#the-recipe--what-the-progress-report-measured)
- **18 encoders** — MegaLoc, Game4Loc, AnyLoc, InfoGeo, DINOv3, DenseUAV ViT and more (`avl/models/`)
- **DenseUAV ViT** — 512-dim cross-view descriptor trained for UAV-to-satellite retrieval
- **MixVPR** — strong general-purpose VPR baseline, 4096-dim descriptors
- **CosPlace** — strong alternative via PyTorch Hub (`ResNet101`, 2048-dim)
- **FAISS hashing**
  - `flat` — exact search and the default for DenseUAV-sized galleries
  - `hnsw` — strong recall/latency for much larger databases
  - `ivfpq` — product-quantized compressed index for large-scale maps
- **Geometric re-ranking** (`avl/rerank.py`, off by default) — verifies the top-N retrieval
  candidates with local features + RANSAC and re-orders them by inlier count. See
  [Geometric re-ranking](#geometric-re-ranking).
- **Geo fusion** — sphere-aware weighted average of the five strongest unique locations
- **Trajectory mode** (`avl/nav/`, `scripts/visloc_traj.py`, Benchmark Console "Trajectory"/"Studio" tabs) — fuses the per-frame VPR pose with a simulated IMU through a 15-state error-state Kalman filter: continuous pose between fixes, χ²-gated outlier rejection, and coasting through VPR dropouts. See `docs/VisualInertial_AVL_Design.md`. The same tab can instead fuse VPR with **real frame-to-frame visual odometry** (`avl/nav/sparse_vo.py`, `scripts/visloc_vo.py`) through a hybrid particle-filter/Kalman filter or the other `avl/nav/vo_fusion.py` filters (`--filter hybrid|pf|kf_reanchor|kf_window|pgo`). Results: `docs/VO_AVL_Fusion_Experiment.md`.

The DenseUAV ViT checkpoint is downloaded from
[`Bancie/UAV-Self-Positioning-23M-ZCN`](https://huggingface.co/Bancie/UAV-Self-Positioning-23M-ZCN)
on first use and then loaded from the local Hugging Face cache.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU FAISS (optional, faster large-scale search):

```bash
pip install faiss-gpu
```

## Reference metadata format

CSV with required columns `image_path`, `latitude`, `longitude`:

```csv
image_path,latitude,longitude,altitude_m,heading_deg,image_id
data/refs/img001.jpg,48.8584,2.2945,50.0,180.0,img001
```

## Quick start (synthetic demo)

```bash
python scripts/create_examples.py

python scripts/build_index.py \
  --metadata examples/references.csv \
  --base-dir . \
  --output artifacts/demo_index

python scripts/localize.py \
  --index artifacts/demo_index \
  --query examples/queries/query.jpg
```

## Benchmark KPIs

Use `scripts/benchmark_kpis.py` to compare neural encoders and FAISS search algorithms.

```bash
python scripts/benchmark_kpis.py \
  --metadata examples/references.csv \
  --base-dir . \
  --query examples/queries/query.jpg \
  --model-config mixvpr:4096 \
  --model-config mixvpr:512 \
  --index-types hnsw flat ivfpq \
  --top-k 5 \
  --repeats 5 \
  --output-json artifacts/kpis.json \
  --output-csv artifacts/kpis.csv
```

Main KPIs:

| KPI | Meaning |
|-----|---------|
| `reference_encode_s` | Offline time to compute NN descriptors for all reference images |
| `reference_encode_ms_per_image` | Offline descriptor throughput per reference image |
| `faiss_build_s` | Offline time to build the FAISS index from descriptors |
| `query_encode_ms_*` | Online time to compute the query image descriptor |
| `search_ms_*` | Online FAISS nearest-neighbor search time |
| `encode_plus_search_ms_mean` | Mean online descriptor + search latency |
| `faiss_index_size_mb` | Serialized FAISS index size |
| `recall_at_1`, `recall_at_5`, `recall_at_10` | Retrieval quality when query ground-truth IDs are provided |
| `top1_error_m`, `fused_error_m` | Optional localization error when query ground truth lat/lon are provided |

For quality KPIs, pass a query CSV:

```csv
image_path,latitude,longitude,expected_image_id
examples/queries/query.jpg,48.85854,2.29464,ref_04
```

```bash
python scripts/benchmark_kpis.py \
  --metadata examples/references.csv \
  --base-dir . \
  --query-metadata examples/query_ground_truth.csv \
  --query-base-dir . \
  --model-config mixvpr:4096 \
  --index-types hnsw flat \
  --recall-k 1 5 10
```

## DenseUAV Testing Dataset

DenseUAV is a low-altitude UAV self-positioning dataset with UAV-view and satellite-view imagery. Its metadata
includes GPS text files in the form `path latitude longitude height`, which can be converted to this project's CSV
format.

Download and extract DenseUAV from Hugging Face:

```bash
python scripts/download_denseuav.py \
  --output-dir data/DenseUAV \
  --download-dir data/downloads
```

Convert DenseUAV metadata to AVL CSV files:

```bash
python scripts/prepare_denseuav.py \
  --dense-root data/DenseUAV \
  --output-dir data/denseuav_avl \
  --reference-split gallery \
  --reference-view satellite \
  --query-view drone
```

For a fast smoke test, prepare a smaller subset:

```bash
python scripts/prepare_denseuav.py \
  --dense-root data/DenseUAV \
  --output-dir data/denseuav_avl_smoke \
  --reference-split gallery \
  --reference-view satellite \
  --query-view drone \
  --limit-references 200 \
  --limit-queries 20
```

Run KPIs on DenseUAV:

```bash
python scripts/benchmark_kpis.py \
  --metadata data/denseuav_avl/references_gallery_satellite.csv \
  --base-dir / \
  --query-metadata data/denseuav_avl/queries_test_drone.csv \
  --query-base-dir / \
  --model-config denseuav-vit:512 \
  --index-types hnsw flat \
  --top-k 5 \
  --query-rotations 4 \
  --repeats 3 \
  --output-json artifacts/denseuav_kpis.json \
  --output-csv artifacts/denseuav_kpis.csv
```

`references_gallery_satellite.csv` is the searchable database. Do not use
`queries_test_drone.csv` as `--metadata`; that file contains evaluation queries,
not satellite references. Four-orientation query search is enabled by default
for aerial imagery to handle unknown UAV heading.

## Localization and KPI Console

The project includes a lightweight native Ubuntu GUI with two separate workflows. The GUI itself does not import
Torch, FAISS, or the neural models; it starts the heavy pipeline commands as separate processes so the interface
stays responsive.

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_desktop_app.py
```

The console provides:

| Area | Purpose |
|------|---------|
| Test & Visualize | Build the reference index once, then keep the model and FAISS index loaded in a persistent online engine for subsequent query images |
| Batch KPI | Select a multiple-query CSV instead of a single image and report feature speed, search speed, total online latency, and Recall@1/5/10 |
| Satellite reference CSV | Choose the geo-tagged satellite gallery CSV, never the query CSV |
| Reusable offline index | Stores `index.faiss`, metadata, and offline timing so later query-image tests do not recompute reference features |
| Geometric re-ranking | Toggle and configure the verification stage. Applies to both workflows; see [Geometric re-ranking](#geometric-re-ranking) |
| Localization result | Preview the selected query, fused estimated latitude/longitude, and ranked Top-K reference images |
| Load KPI JSON | Display a previously generated `benchmark_kpis.py` JSON report without running the model |

The interactive workflow is:

1. Select the reference CSV and pipeline configuration.
2. Click **Build offline index** once.
3. The console starts and warms the persistent online engine.
4. Select a query image and click **Run online query**.
5. Select another query image and click **Run online query** again. The loaded model and index are reused.

The first engine startup includes Python/Torch imports, model loading, index loading, and one warm-up pass.
It happens asynchronously after opening an existing index or after building a new one. Once the status reads
**ENGINE READY**, each interactive click performs exactly one feature extraction and one FAISS search. The
**Online total** card shows visible GUI latency, while its detail separates worker end-to-end time from the
feature-plus-search pipeline time. The **Benchmark repeats** setting applies only to Batch KPI mode.

Re-ranking settings are sent with each query rather than baked into the engine, so toggling the checkbox or
moving a threshold takes effect on the next click without reloading the model or the index. The **Re-rank**
card reports the stage's latency and how many candidates passed verification, and each match card shows its
inlier count and how far verification moved it (`↑ was #6`).

The batch CSV must contain:

```csv
image_path,latitude,longitude,expected_image_id
path/to/query_001.jpg,30.1,120.2,location_001
```

`latitude` and `longitude` are optional for recall, but `expected_image_id` is required for Recall@1/5/10.

Build a Linux executable:

```bash
bash scripts/build_desktop_app.sh
./dist/AVL-Mission-Console/AVL-Mission-Console
```

## Benchmark Console

The main GUI (`avl/model_bench_gui.py`, components in `avl/console_widgets.py`): pick a map and
a flight, build a recipe, run it, and read the results next to what the report measured.

```bash
python scripts/run_model_bench.py      # or the desktop launcher (below)
```

The window is laid out like an operations console:

- **Top bar** — the map and flight pickers (chosen independently, so cross-dataset runs are
  allowed), a pill with the recipe the next run uses, **Run** / **Stop**, and the inspector toggle.
  A 2 px line under it shows progress.
- **Rail (left)** — the pages below.
- **Recipe inspector (right)** — what the selected flight carries (heading ✓, altitude ✓,
  region, tile size); a preset (Baseline, Recommended, Best accuracy); the encoder — one, an
  **ensemble** of the ticked ones, or a **comparison** (one run each); and every measured step as
  a switch with its effect from the report and its settings. Steps the flight cannot feed are
  greyed out with the reason.
- **Status bar** — state, the running job with its ETA, saved runs, and the **Output** panel
  with the subprocess log.

| Page | Purpose |
|------|---------|
| Overview | KPI tiles for the current recipe on the selected flight — top-1 and fused within 100 m, median error, each with its change against the Baseline preset, and the projected on-board latency on a Jetson Orin Nano Super (report §9, a projection). The pipeline as a flow (switched-off stages hollow, the running stage glowing), the progress ladder, and the latest runs. |
| Runs | Every saved run: encoder(s), the steps it used, top-1 within 100 m with an in-cell bar, fused, top-5, median error, chance, GMAC per frame, speed. Filter by map or text; click a row to inspect it. |
| Findings | The report's step-by-step ladder rebuilt from *your* runs on the selected map and flight, with **Run missing steps**; below, one card per method: idea, what this repository does, measured numbers, cost, needs, **Use in recipe**. |
| Inspect | One frame at a time — the frame, its position against the truth, how the recipe prepared it (rotation, crop, AGL) and its five best tiles. From a saved run, or any image (heading and altitude filled from the flight CSV when it is one of its frames). |
| Models | The 18 encoders: GMAC, raw accuracy on r05 / r10 / r06, descriptor, training data. |
| Flight | Fuse the per-frame fix with a **simulated IMU** through a 15-state error-state Kalman filter (`scripts/visloc_traj.py`). See `docs/VisualInertial_AVL_Design.md`. |
| Studio | Draw a route over the region's satellite mosaic and fly it with the same filter. |

Install a double-clickable launcher (applications menu plus a desktop icon):

```bash
bash scripts/install_launcher.sh
```

The launcher runs `scripts/run-model-bench.sh`, which uses the project's `.venv` interpreter. That matters
because the console starts `scripts/visloc_eval.py` / `scripts/visloc_traj.py` with `sys.executable`, so it
needs a real environment with Torch and FAISS rather than a frozen bundle. Launch failures are written to
`artifacts/model_bench_launcher.log`.

## Production usage

### 1. Build index from your geo-tagged map

The index stores its recipe. By default it is built with the recommended preset; give the
camera constant of your lens (`k = 2·tan(HFOV/2)`) for the altitude crop:

```bash
python scripts/build_index.py \
  --metadata data/visloc_avl/r05_t250/references.csv \
  --output artifacts/index_r05 \
  --camera-k 0.971                  # --preset accuracy for the ensemble
```

`--preset none --model mixvpr --descriptor-dim 4096 --index-type hnsw` builds the historical
single-encoder index with no recipe. For very large databases (millions of tiles) add
`--index-type ivfpq`.

### 2. Localize a UAV frame

The frame's telemetry goes on the command line; a step whose input is missing is skipped
(no heading → four-rotation search, no altitude → no crop, no prior σ → whole map):

```bash
python scripts/localize.py \
  --index artifacts/index_r05 \
  --query data/UAVVisLoc/05/drone/05_0001.JPG \
  --heading-deg 91.1 --altitude-asl 2315 \
  --prior-lat 24.6659 --prior-lon 102.3413 --prior-sigma 100 \
  --json-out outputs/localization.json
```

### 3. Python API

```python
from pathlib import Path
from avl import AVLLocalizer
from avl.config import AVLConfig
from avl.pipeline import QueryTelemetry

config = AVLConfig.from_preset("recommended", camera_k=0.971)
localizer = AVLLocalizer(config)
localizer.build_index(Path("data/visloc_avl/r05_t250/references.csv"), Path("artifacts/index"))

localizer.load_index(Path("artifacts/index"))      # the recipe comes with the index
for frame, imu in flight:                            # in flight order: the flight mean is causal
    result = localizer.localize(frame, telemetry=QueryTelemetry(
        heading_deg=imu.heading, altitude_asl_m=imu.baro_alt,
        prior_lat=imu.lat, prior_lon=imu.lon, prior_sigma_m=imu.sigma))
    print(result.pose.latitude, result.pose.longitude, result.confidence, result.plan)
localizer.reset_flight()                             # before the next flight
```

## Model choices

| Model | Descriptor dim | Best for |
|-------|----------------|----------|
| `mixvpr` (default) | 4096 | Highest VPR accuracy |
| `mixvpr` | 512 | Faster indexing, lower memory |
| `cosplace` | 2048 | Alternative backbone, PyTorch Hub |
| `eigenplaces` | 2048 | Ground-level VPR, CosPlace successor (PyTorch Hub) |

The Model Benchmark Console additionally offers cross-view UAV↔satellite encoders —
`denseuav-vit` (512), `game4loc` (768), `sample4geo` (1024) — the training-free
AnyLoc family (`anyloc-lite` DINOv2 ViT-B + final tokens, `anyloc-l` ViT-L + value
facet, `anyloc-g` ViT-G + value facet — the paper's method; VLAD vocabulary and PCA
dimension fitted per map) and `dinov2` (768, control). `anyloc-g` is ~6 s/image on
CPU — use it sparingly.

```bash
python scripts/build_index.py --model cosplace --descriptor-dim 2048 --cosplace-backbone ResNet101 ...
```

## Index tuning

| Index | When to use |
|-------|-------------|
| `hnsw` | < 1M references, best accuracy/speed trade-off |
| `ivfpq` | Large maps, memory-constrained deployment |
| `flat` | Small maps, debugging |

## Geometric re-ranking

A global descriptor collapses a whole tile into one vector, so on aerial imagery —
repetitive fields, roof grids, tree canopy — the correct reference often lands at
rank 3-10 rather than rank 1. Every aerial VPR study we follow treats retrieval as a
*candidate generator* and settles the ranking with local features. `avl/rerank.py`
is that second stage: it matches the query against each of the top-N candidates,
runs RANSAC on the correspondences, and re-orders by inlier count.

Turn it on with the **Geometric re-ranking** checkbox in the Localization Console, or with
`--rerank` on `scripts/benchmark_kpis.py`, `scripts/visloc_eval.py`, `scripts/visloc_query.py`
and `scripts/localize.py`. From Python it is `AVLConfig(rerank_enabled=True, …)`, honoured by
`AVLLocalizer`. It is **off by default** — existing runs are bit-identical without it.

`scripts/visloc_eval.py` additionally reports ms/query, how many candidates verified on
average, how many queries verified nothing, and how often verification promoted a new
top-1 — the four numbers that say whether the stage is earning its latency.

| Backend | Extra dependency | Notes |
|---------|------------------|-------|
| `sift-ransac` (default) | — | Rotation and scale invariant, FLANN-matched. ~35 ms/candidate |
| `orb-ransac` | — | Faster, noticeably weaker |
| `akaze-ransac` | — | Cheapest of the three, good on low-texture terrain |
| `superpoint-lightglue` | `pip install lightglue` | Strongest pairing in the aerial VPR literature |
| `loftr` | `pip install kornia` | Detector-free dense matching; best across viewpoint change, slowest |

Install the optional backends together with `pip install -e '.[rerank]'`.

### Settings

| Setting | Default | Meaning |
|---------|---------|---------|
| Candidates | 10 | Short-list depth handed to verification. Deeper recovers more true matches that retrieval ranked low, at one match per tile |
| Min inliers | 12 | RANSAC inliers before a candidate counts as *verified* |
| Geometry weight (`blend`) | 0.5 | 0 keeps the descriptor order, 1 ranks purely on geometry |
| Max features | 2048 | Keypoint budget per image |

### How it behaves

Verification is a **gate, not a vote**. Candidates that pass rank above those that do
not, and a rejected candidate contributes nothing to the fused position. When *nothing*
verifies — a cross-domain pair local features cannot bridge — the order and the pose
are exactly what retrieval produced. So the stage can improve a result or abstain from
it, but it cannot scramble a good ranking.

RANSAC will happily fit a homography to ~70 random correspondences, so every fit is
checked for degeneracy first: the query outline must map to a convex quad of comparable
area. Without that check the fake inliers from a collapsed warp are indistinguishable
from a true match's, and the stage ranks on noise.

### When it helps, and when it does not

Re-ranking pays off when the query and the references are the **same kind of imagery**.
On an aerial-to-aerial pair (satellite tile vs. a rotated, scaled, brightened crop of
the same area) SIFT lifts the true tile from rank 6 to rank 1 with 340 inliers against
0 for the rejects.

It does **not** help across a drone-to-satellite domain gap. Measured on both bundled
datasets with `sift-ransac`, 10 candidates:

| Dataset | Query vs. reference | Verified | Effect |
|---|---|---|---|
| `data/denseuav_avl_smoke` | 1440×1080 drone photo vs. 512×512 satellite tile | 0/10 on every query | R@1 unchanged at 0.60, +490 ms/query |
| `data/visloc_avl/r10_t200` | 3000×2000 drone photo @772 m vs. 512×512 basemap tile | 0/10 on **144/144** queries | every metric unchanged, +422 ms/query |

In both cases the failure is real, not a threshold to tune: even the tile 45 m from the
query produces only ~40 ratio-test matches, and every homography RANSAC fits to them is
degenerate. `orb-ransac` and `akaze-ransac` behave the same way.

Check the **Re-rank** metric box first. If it reports `0/N verified`, local features are
not bridging your domain gap and the stage is pure latency — turn it off. The learned
backends (`superpoint-lightglue`, `loftr`) are the ones the aerial VPR literature uses
for exactly this gap and are the next thing to try, but they are untested here.

## Scale from altitude (AGL)

A drone frame covers `k × AGL` metres of ground, where `k = 2·tan(HFOV/2)` and AGL is the height
above the *ground*. When that is far from the reference tile size, the encoder compares the right
place at the wrong zoom. Measured on UAV-VisLoc r11 (frames ≈ 1.9× the tile), correcting it lifts
MegaLoc top-1 within 100 m from 26 % to 65 %, at one encode per frame instead of the blind
three-scale search's three. Full study: [docs/Scale_Normalization_Report.md](docs/Scale_Normalization_Report.md).

```bash
pip install -e '.[terrain]'      # tifffile + imagecodecs, to read the DEM
python scripts/visloc_eval.py --refs <map>/references.csv --queries <map>/queries.csv \
    --model megaloc --rotations 1 --north-align --yaw-sign -1 --north-align-mode auto \
    --scale-from-agl --camera-k 0.96 --agl-gate 1.25
```

- AGL = flight-log altitude minus the Copernicus GLO-30 terrain, downloaded once per 1° cell
  into `data/dem/` (`avl/terrain.py`). With `--prior-sigma`, the terrain is read at the
  simulated prior position instead of the truth.
- `--camera-k` comes from the lens datasheet, or from `scripts/visloc_scale.py calibrate`. Pixel
  count does not identify the lens: UAV-VisLoc's 3000 × 2000 frames come from two different lenses.
- `--agl-gate 1.25` crops only frames that cover more than 1.25× a tile. Smaller mismatches keep
  the whole frame, whose context is worth more.
- Better still, build the map at the frame's scale:
  `python scripts/visloc_scale.py plan --regions <r> --base-tiles <map> --strides <m> --prepare`
  picks the tile size from AGL with no ground truth.

## UAV tips

1. Restrict search to a geographic tile when a coarse prior exists (pre-filter your CSV before indexing).
2. Use multi-frame temporal filtering in your flight stack (not included here).
3. Match reference viewpoint: nadir UAV frames work best against nadir/satellite references.
4. First run downloads MixVPR weights (~150 MB) to `~/.cache/avl/mixvpr/`.

## Project layout

```
avl/
  pipeline.py     # the recipe: presets, heading / altitude plan per frame, telemetry, ladder
  ensemble.py     # encoder ensembles (equal-vote concatenation) and per-member caches
  centering.py    # domain centering (map / flight mean), block-wise for ensembles
  findings.py     # the progress report's measured findings, as data for the console
  encoder.py      # one encoder (18 models in avl/models/)
  index.py        # FAISS flat / HNSW / IVF-PQ index, stores the recipe, windowed search
  localizer.py    # End-to-end AVL with telemetry (heading, altitude, IMU prior)
  model_bench_gui.py, console_widgets.py   # the benchmark console
  geo.py          # Haversine + weighted fusion
  rerank.py       # geometric verification of the retrieval short-list
  retrieval.py    # shared single-frame retrieval core
  terrain.py      # Copernicus GLO-30 DEM: terrain height, height above ground (AGL)
  scale.py        # camera footprint (k, HFOV, GSD) -> per-frame query crop for scale normalisation
  nav/            # trajectory mode: IMU sim, strapdown INS, error-state EKF, mosaic raster,
                  #   sparse visual odometry, VO+AVL fusion filters (KF / PF / hybrid / pose graph)
scripts/
  build_index.py
  localize.py
  visloc_eval.py  # benchmark one recipe on a prepared map + flight
  visloc_query.py # localize one image with a recipe
  check_localizer_parity.py  # does AVLLocalizer reproduce visloc_eval frame by frame?
  create_examples.py
  visloc_traj.py  # time-ordered VPR + IMU (ESKF) or VPR + visual odometry fused trajectory
  visloc_vo.py    # frame-to-frame visual odometry on a prepared flight, error vs ground truth
  vo_fusion_experiment.py  # offline comparison of VO+AVL fusion methods on cached descriptors
  nav_selfcheck.py
  visloc_scale.py # AGL / camera calibration / scale-normalisation experiment (report: docs/Scale_Normalization_Report.md)
```

## References

- [MixVPR](https://arxiv.org/abs/2303.02190) — Amar Ali-bey et al., WACV 2023
- [CosPlace](https://arxiv.org/abs/2204.02287) — Berton et al., CVPR 2022
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
