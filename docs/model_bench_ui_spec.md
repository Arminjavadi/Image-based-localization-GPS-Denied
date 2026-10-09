# Model Benchmark Console — UI design brief & API reference

A hand-off document for redesigning the **Model Benchmark Console** GUI
(`avl/model_bench_gui.py`, launched by `scripts/run_model_bench.py`).
It describes what the app does, every screen and control, the data it consumes and
produces (the "API"), the runtime states, and the visual direction to keep.

---

## 1. What the app is

A desktop console (PySide6 / Qt Widgets, Linux-first) for evaluating **visual place
recognition (VPR) encoders** for GPS-denied UAV localization. The operator picks a
reference map, a set of query images, and one or more neural encoders, and the app
reports **where each query image is on the map** (latitude / longitude) plus how
close that estimate is to ground truth.

The GUI is a thin shell: it never imports Torch / FAISS. It shells out to two Python
scripts as subprocesses and renders their JSON / CSV output:

| Script | Role | Output |
|---|---|---|
| `scripts/visloc_eval.py` | Evaluate an encoder over a whole query set | `<tag>_summary.json` + `<tag>_per_query.csv` in `artifacts/visloc/` |
| `scripts/visloc_query.py` | Localize **one** image | JSON on stdout (`RESULT_JSON …`) and to `--out` |

Both share the same retrieval core (`avl/retrieval.py`): encode the query in 4
rotations (0/90/180/270°), cosine-match against every reference tile, keep the best
orientation per tile, de-duplicate to one hit per physical location, take the top-K,
then **weighted geo-fusion** of the top-5 into a single lat/lon.

### Primary users
UAV localization researchers comparing encoders across datasets. They care about:
localization error in metres, recall within distance thresholds, and *why* a
particular frame localized well or badly (looking at the actual retrieved tiles).

---

## 2. Information architecture

```
┌ Left control panel (always visible) ─────────┐  ┌ Right: tabbed workspace ─────────────┐
│  Reference database   (combo)                │  │  Tab 1  Comparison                   │
│  Query set            (combo)                │  │  Tab 2  Inspector                    │
│  Active encoder       (combo)                │  │  Tab 3  Single image                 │
│  Encoders to benchmark(checklist)            │  │                                     │
│  Protocol             (4 fields)             │  │                                     │
│  Run benchmark / Stop / Reload  (buttons)    │  │                                     │
│  Status pill                                 │  │                                     │
└──────────────────────────────────────────────┘  └─────────────────────────────────────┘
```

The four selectors are **independent**. Any encoder can be run against any reference
database with any query set, including cross-dataset combinations (e.g. the DenseUAV
satellite map searched with UAV-VisLoc drone photos).

---

## 3. Left control panel — controls

| Control | Type | Values / behaviour |
|---|---|---|
| **Reference database** | dropdown | Auto-discovered `references*.csv` under `data/`. Plus **Browse CSV…**, **Rescan**. This is the searchable map. |
| **Query set** | dropdown | Auto-discovered `queries*.csv`. Plus **Browse CSV…** and **Pair with reference folder** (opt-in: snap to the CSV sitting beside the selected database). |
| **Active encoder** | dropdown | One of the 6 models (below). Drives the **Single image** and **Inspector** tabs. |
| **Encoders to benchmark** | checklist | Same 6 models, multi-select. Drives the **Comparison** tab only. |
| **Protocol → Query rotations** | dropdown | `4 (heading unknown)` \| `1 (heading known)` |
| **Protocol → Query crop** | dropdown | `square` \| `none` (square = centre-crop query to 1:1 before encoding) |
| **Protocol → Max queries** | integer 1–10000, default 144 | cap on query-set size per run |
| **Protocol → Query stride** | integer 1–100, default 1 | evaluate every Nth query |
| **Run benchmark** | primary button | queue one `visloc_eval.py` per ticked encoder |
| **Stop** | button | kill the running subprocess, clear the queue |
| **Reload saved results** | button | rescan `artifacts/visloc/` |
| **Status pill** | text label | `READY` / `RUNNING <tag>` / `LOCALIZING <file>` / `STOPPED` — colour: green idle, amber busy |

### The 6 encoders

| id | Label shown | One-liner |
|---|---|---|
| `denseuav-vit` | DenseUAV ViT | UAV↔satellite, trained on 14 Hangzhou campuses (512-d) |
| `game4loc` | Game4Loc | UAV↔satellite, cross-area split, GTA-UAV (768-d) |
| `anyloc-lite` | AnyLoc-lite | DINOv2 + VLAD, vocabulary fitted on the selected map |
| `dinov2` | DINOv2 | control: no cross-view training (768-d) |
| `mixvpr` | MixVPR | ground-level VPR baseline (4096-d) |
| `cosplace` | CosPlace | ground-level VPR baseline (2048-d) |

---

## 4. Tab 1 — Comparison

**Purpose:** run the ticked encoders over the selected database + query set and read
localization metrics side by side. One table row per encoder run.

**Components**
- **Metrics table** (one row per finished run):

  | Column | Source key | Notes |
  |---|---|---|
  | Run | `tag` | e.g. `r10_t200__denseuav-vit` — never truncate |
  | Tiles | `n_refs` | |
  | Queries | `n_queries` | |
  | Top-1 median | `top1_error_m.median` | metres |
  | Top-1 p95 | `top1_error_m.p95` | metres |
  | <50 m | `recall_top1_within_m["50"]` | % |
  | <100 m | `recall_top1_within_m["100"]` | **headline metric** — bold, colour-coded: >70 % green, >30 % amber, else red |
  | Top-5 <100 m | `recall_topk_within_m["100"]` | % |
  | Chance <100 m | `chance_baseline.random_tile_within_m["100"]` | % — random-guess baseline for context |
  | ms/query | `query_ms_per_image` | |

- **Log pane** — monospaced, ~190 px tall, streams subprocess stdout.

**Run lifecycle:** `Run benchmark` → queue N runs → status `RUNNING <tag>` → each
finishes → table + Inspector run list refresh → `READY` when queue drains.

---

## 5. Tab 2 — Inspector

**Purpose:** per-query drill-down. Pick one evaluated query and see the exact tiles
the encoder retrieved, with coordinates and error.

**Components**
- **Run controls** (new): a group titled *"Evaluate the selected encoder + reference
  database + query set"* with:
  - **Run inspector on selected DB + query set** (primary button) — evaluates the
    **Active encoder** against the current database + query set (using the Protocol
    box). Reuses a saved `<tag>_per_query.csv` if present; otherwise runs
    `visloc_eval.py`. On finish it auto-selects the new run and stays on this tab.
  - **force re-run** (checkbox) — recompute even if a saved result exists.
- **Run** dropdown — every finished run in `artifacts/visloc/` (`tag`).
- **Split view:**
  - *Left* — **query list table**: columns `Query` (id), `Top-1 err`, `Best top-5`,
    `Score`. `Top-1 err` cell is green ≤100 m, red otherwise. Row select drives the
    right panel.
  - *Right* — **Retrieval panel** (see §7), inside a vertical scroll area.

---

## 6. Tab 3 — Single image

**Purpose:** localize an arbitrary image that is not part of any prepared query set
(a new drone frame, a screenshot, etc.).

**Components**
- **Image** — text field + **Browse…** (jpg/png/tif).
- **Encoder** — read-only label mirroring the left-panel **Active encoder**.
- **Localize image** — primary button → runs `visloc_query.py`.
- **Hint line** — reports which map is searched and whether ground truth was found
  for this image in the current query set (so error can be shown).
- **Retrieval panel** (see §7) — identical rendering to the Inspector.

If the picked image also appears in the selected query set, its ground-truth lat/lon
is passed as `--gt-lat/--gt-lon` so per-match and fused error are filled in.

---

## 7. The Retrieval panel (shared by Inspector + Single image)

The core result visual. Must render identically in both tabs so a benchmark row and
a live lookup never look different.

Sections, top to bottom:

1. **Query frame** — image preview, ~300×200, "Select a query" placeholder.
2. **Position** — 2-column grid:
   | Row | Value |
   |---|---|
   | Ground truth | `lat, lon` (6 dp) or `-` |
   | Fused estimate (top-5) | `lat, lon` |
   | Top-1 tile | `lat, lon   (tile_id, score 0.xxx)` |
   | Error | `top-1 NNN m · fused NNN m · best of top-5 NNN m · confidence 0.xxx · spread NNN m` |
3. **Retrieved tiles (rank 1 → 5)** — a horizontal row of 5 thumbnails (~96×96),
   each with a caption: `#k · NNN m` then the tile's `lat` / `lon` on two lines.
4. **Top-K table** — columns `#`, `Tile id`, `Latitude`, `Longitude`, `Score`,
   `Error`. The `Error` cell is green ≤100 m, red otherwise. Sized to its rows (no
   scrollbar for 5 results).

Empty state resets every field to `-` and clears all previews.

---

## 8. Data contracts (the "API")

### 8.1 Reference / query CSV (input)

Required columns: `image_path`, `latitude`, `longitude`.
Optional: `image_id` (falls back to `row_<i>`), `altitude_m`, `heading_deg`, `yaw_deg`.
`image_path` may be absolute or relative to the CSV's own folder.

```csv
image_path,latitude,longitude,altitude_m,heading_deg,image_id
10/drone/10_0013.JPG,30.51234,114.30021,120.0,,10_0013
```

### 8.2 `visloc_query.py` result JSON — one image

```jsonc
{
  "query_image": "data/UAVVisLoc/10/drone/10_0013.JPG",
  "model": "denseuav-vit",
  "refs_csv": "data/visloc_avl/r10_t200/references.csv",
  "n_refs": 812,
  "rotations": 4,
  "query_crop": "square",
  "matches": [
    {
      "rank": 1,
      "image_id": "r10_t200_0417",
      "latitude": 30.5127,
      "longitude": 114.3010,
      "score": 0.8123,          // cosine similarity, 0..1
      "image_path": "data/visloc_avl/r10_t200/tiles/0417.png",
      "error_m": 63.0           // null when no ground truth
    }
    // … up to top_k (default 5), one per unique location
  ],
  "fused": { "latitude": 30.5126, "longitude": 114.3009, "altitude_m": null },
  "spread_m": 88.4,             // mean distance of the 5 tiles from the fused point
  "confidence": 0.71,           // top1_score * exp(-spread_m / 500)
  "ground_truth": { "latitude": 30.5125, "longitude": 114.3008 }, // or null
  "fused_error_m": 41.0,        // or null
  "timings": { "query_encode_ms": 190.0 }
}
```

### 8.3 `<tag>_per_query.csv` — one row per query (Inspector source)

| Column | Meaning |
|---|---|
| `query_id` | query image id |
| `gt_lat`, `gt_lon` | ground truth |
| `top1_id` | id of the rank-1 tile |
| `top1_score`, `top2_score`, `score_margin` | rank-1 / rank-2 cosine, and their gap |
| `topk_ids` | `\|`-joined tile ids, rank 1→K |
| `topk_errors_m` | `\|`-joined per-tile distance to GT, metres |
| `topk_scores` | `\|`-joined cosine scores |
| `top1_error_m` | distance of rank-1 tile to GT |
| `best_topk_error_m` | smallest error among the top-K |
| `fused_error_m` | distance of the fused estimate to GT |
| `fused_lat`, `fused_lon` | fused estimate |
| `spread_m` | mean tile spread around the fused point |
| `confidence` | as in 8.2 |

### 8.4 `<tag>_summary.json` — aggregate (Comparison source)

```jsonc
{
  "tag": "r10_t200__denseuav-vit",
  "model": "denseuav-vit",
  "refs_csv": "…/references.csv",
  "queries_csv": "…/queries.csv",
  "n_refs": 812,
  "n_queries": 144,
  "rotations": 4,
  "query_stride": 1,
  "max_queries": 144,
  "north_align": false,
  "query_crop": "square",
  "top_k": 5,
  "reference_encode_s": 41.2,
  "query_total_s": 27.4,
  "query_ms_per_image": 190.3,

  // every *_error_m / *_score block is { mean, median, p95, min, max }
  "top1_error_m":      { "mean": …, "median": …, "p95": …, "min": …, "max": … },
  "fused_error_m":     { … },
  "best_topk_error_m": { … },
  "top1_score":        { … },
  "score_margin":      { … },

  // fraction (0..1) of queries within each distance, keys "25","50","100","200","500","1000"
  "recall_top1_within_m": { "25": 0.11, "50": 0.34, "100": 0.62, … },
  "recall_topk_within_m": { … },   // best of top-5
  "fused_within_m":       { … },

  "chance_baseline": {
    "random_tile_mean_m": …,
    "random_tile_median_m": …,
    "oracle_best_tile_median_m": …,      // error of the single closest tile in the map
    "random_tile_within_m": { "25": …, "50": …, "100": …, … }
  }
}
```

### 8.5 Datasets currently on disk (for realistic mock data)
- `data/denseuav_avl/` — `references_gallery_satellite.csv`, `queries_test_drone.csv`
- `data/visloc_avl/r10_t100 … r10_t600`, `r06_t200` — `references.csv`, `queries.csv`
  (`t###` = reference tile ground size in metres)

Typical scale: 200–3000 reference tiles, 20–150 queries, errors from ~20 m (good) to
several km (chance), scores 0.4–0.9.

---

## 9. Runtime states

| State | Trigger | UI |
|---|---|---|
| `READY` | idle | green pill, run buttons enabled |
| `RUNNING <tag>` | benchmark / inspector run in progress | amber pill, `Run`/`Run inspector` disabled, `Stop` enabled, log streaming |
| `LOCALIZING <file>` | single-image run | amber pill, `Localize` disabled |
| `TRAJECTORY <tag>` | Trajectory-tab run (`visloc_traj.py`) | amber pill, `Run trajectory` + `Run fused` disabled, `Stop` enabled |
| `STUDIO <tag>` | Studio-tab run (`visloc_traj.py --frame-subset`) | amber pill, both trajectory run buttons disabled |
| `STOPPED` | user hit Stop | green-ish pill, queue cleared |
| result-load failure | bad JSON / non-zero exit | error text in the hint line + log |

Only one subprocess runs at a time; Comparison and Inspector share the queue and
must not launch concurrently. The Trajectory and Studio tabs share one process
handle (`self.traj_process`) and are also guarded by the single-run check.

---

## 10. Visual direction (keep the identity, improve the execution)

Current theme is a dark analyst console. Preserve the intent; a redesign may refine
spacing, hierarchy, typography, and the results visual.

| Token | Current value | Role |
|---|---|---|
| app background | `#10141c` | |
| panel / card | `#161c27` / `#141a24` | |
| border | `#232c3b` / `#2e3a4e` | |
| primary action | `#2f6df0` | Run buttons, selected tab/row |
| text strong / body / hint | `#f2f6ff` / `#d8e0ee` / `#8fa0bb` | |
| good / warn / bad | `#7ee0a8` / `#f0c04a` / `#f07a7a` | error & recall colour-coding |
| monospace | coordinates, ids, scores, log | |

**Priorities for the redesign**
1. Make the **four independent selectors** legible at a glance — it must be obvious
   that database, query set, and encoder are chosen separately, and which encoder
   feeds which tab.
2. The **Retrieval panel** is the payoff screen: query frame vs. 5 retrieved tiles,
   with a small map / coordinate readout, error and confidence prominent. A mini map
   plotting GT, fused estimate, and the 5 tiles would be a strong addition.
3. Comparison table: the `<100 m` recall column is the headline — let it dominate;
   keep run tags fully visible.
4. Dense but calm: researchers scan many numbers; alignment and restrained colour
   matter more than decoration.
5. Responsive from ~1200 px to full screen; the left panel is fixed-width (~380 px),
   the workspace flexes.

**Screens to mock:** (a) left panel, (b) Comparison with 3–4 runs, (c) Inspector
split view with a query selected, (d) Single image with a result, (e) empty / first-run
state, (f) running state with the log.

---

## 11. Open questions for the designer
- Should the three tabs stay tabs, or become one adaptive view driven by the selectors?
- Where should a map view live — inside the Retrieval panel, or as its own pane?
- Is the Comparison log worth its vertical space, or should it collapse to a drawer?
- Multi-run compare: table only, or add small-multiples charts (recall-vs-distance curves)?

---

## 12. Trajectory & Studio tabs — VPR + IMU fusion

Added after the redesign. Full rationale, equations, and the phased build log are
in [`VisualInertial_AVL_Design.md`](VisualInertial_AVL_Design.md); this section is
the UI + data-contract summary.

### 12.1 What they do

The base tabs localise each query **independently**. These two run a query set as a
**time-ordered trajectory** and fuse the per-frame VPR pose with a **synthetic
IMU** through a 15-state error-state Kalman filter (`avl/nav/`, numpy-only). The
payoff is a continuous solution between fixes, χ²-gated rejection of stray VPR
matches, and bounded error through fix dropouts.

| Tab | Role |
|---|---|
| **Trajectory** | Batch: run the whole selected query set, read the three-way **IMU-only / VPR-only / Fused** comparison — a ground-track plot, an error-vs-time plot, a 3×9 metrics table. |
| **Studio** | Draw a route over the region satellite mosaic → snap it to the nearest real drone frames (arc-length order, within a corridor) → run the fused estimator on that subset → overlay GT / IMU-only / fused tracks + VPR fixes on the mosaic. |

Both shell out to **`scripts/visloc_traj.py`** (the GUI never imports Torch), reuse
the left panel's reference database / query set / active encoder / geo-fusion /
rotations / crop, and stream stdout to the shared Comparison-tab log.

### 12.2 "Fusion" control group (Trajectory tab; Studio reads the same widgets)

| Control | Values / default | `visloc_traj.py` flag |
|---|---|---|
| IMU error model | consumer MEMS *(def)* / tactical / none — perfect IMU | `--imu-grade {consumer,tactical,perfect}` |
| IMU noise × | 0.1–10.0, default 1.0 | `--imu-scale` |
| Fusion filter | error-state EKF · IMU *(def)* / hybrid PF→KF / particle filter / Kalman + re-anchor / Kalman + search window / robust pose graph — the last five run on real visual odometry | `--filter {eskf,hybrid,pf,kf_reanchor,kf_window,pgo}`; the VO filters also get `--scores-cache` / `--vo-cache` under `artifacts/visloc/traj_cache/` so switching filter re-runs only the fusion. Studio always sends `eskf` (a drawn route has no images). |
| VPR fix rate | every N frames, 1–20 | `--vpr-every` |
| Fix dropouts | seconds, e.g. `120-180, 300-330` | `--dropout` |
| Initial velocity | known (from GT) *(def)* / zero | `--init-vel` |
| Fuse altitude | off *(def)* | `--fuse-altitude` |
| Real frame timestamps | on *(def)* — source per-frame time from `data/UAVVisLoc/<r>/<r>.csv` | `--region-csv` (omitted for Studio subsets) |
| Frame stride / Max frames | 1 / 120 | `--frame-stride` / `--max-frames` |
| RNG seed | 0 | `--seed` |
| Studio: corridor width | 20–2000 m, default 150 | (snap only) |

### 12.3 `scripts/visloc_traj.py` outputs (in `artifacts/visloc/`)

**`<tag>_traj.csv`** — one row per trajectory frame: `frame, t_s, query_id`,
`gt_{lat,lon,alt}`, `ins_{lat,lon,alt}` (IMU-only), `vpr_{lat,lon}` (`nan` on
non-fix frames), `vpr_conf`, `vpr_spread_m`, `fused_{lat,lon,alt}`,
`err_{ins,vpr,fused}_m`, `along_m`, `cross_m`, `fix_used`, `fix_gated`,
`P_pos_sigma_m`, `vx vy vz`, `bias_a_norm`, `bias_g_norm`.

**`<tag>_traj.json`** — config echo + three metric blocks + plot series:

```jsonc
{
  "tag": "...", "model": "...", "fusion": "cluster", "filter": "eskf",
  "imu": { "grade": "consumer", "scale": 1.0, "rate_hz": 100, "seed": 0 },
  "vpr_every": 1, "fuse_altitude": false, "init_vel": "known",
  "dropouts_s": [[120, 180]],
  "n_frames": 144, "duration_s": 431.0, "path_length_m": 6980.0,
  "query_ms_per_frame": 190.0,
  "origin": { "lat": …, "lon": …, "alt": … },
  "region_mosaic": { "path": "…/satellite05.tif", "lt": [lat,lon], "rb": [lat,lon] },

  // each *_error_m block = { mean, median, p95, min, max }
  "ins_only": { "horiz_error_m": {…}, "rmse_m": …, "final_error_m": …, "drift_rate_pct": … },
  "vpr_only": { "horiz_error_m": {…}, "rmse_m": …, "cep50_m": …, "cep95_m": …,
                "n_frames_with_fix": …, "n_outliers_gt_200m": …, "within_m": {…} },
  "fused":    { "horiz_error_m": {…}, "along_track_m": {…}, "cross_track_m": {…},
                "rmse_m": …, "final_error_m": …, "drift_rate_pct": …,
                "fix_availability": …, "fixes_accepted": …, "fixes_gated": …,
                "mean_nees": …, "within_m": {…} },

  "series": {
    "t_s": [...],
    "gt_en":   [[e,n], ...],   "ins_en":  [[e,n], ...],
    "vpr_en":  [[e,n]|null, ...], "fused_en": [[e,n], ...],
    "err_ins_m": [...], "err_vpr_m": [.. |null ..], "err_fused_m": [...],
    "P_sigma_m": [...], "fix_used": [0|1, ...], "fix_gated": [0|1, ...]
  }
}
```

`series.*_en` are local ENU metres with origin at frame 0; `origin` +
`region_mosaic` convert them back to pixels for the Studio overlay.

### 12.4 Rendering

- **`TrajectoryPlot(QWidget)`** — custom `paintEvent` line plot (no matplotlib /
  pyqtgraph): polylines, dot series, gridlines + tick labels, legend, shaded
  x-bands (dropouts), optional equal-aspect. Draws both the ground track and the
  error-vs-time chart; IMU-only is clipped to 4× the VPR/fused max so the shape
  stays legible.
- **`MosaicView(QGraphicsView)`** (Studio) — pan/zoom mosaic thumbnail; Draw mode
  lays amber dashed vertices; `draw_polyline` / `draw_markers` add overlays. Scene
  coords are thumbnail pixels; overlays convert `geo → full px → × scale` via
  `avl/nav/raster.py::RegionRaster`.
- Palette: GT `#cfd2de`, IMU-only `#f07a7a`, VPR `#f0c04a`, fused `#b5abfc`
  (Nocturne). Metric cells for `p95` / `final` are green `<50 m`, amber `<150 m`,
  red otherwise.

### 12.5 Snap-to-frames

`avl/nav/raster.py::snap_path_to_frames(path_xy, frame_xy, ids, corridor_m)` — the
drawn polyline vertices and every query-frame position are placed in one local ENU
frame; each frame's min distance to the polyline and the arc-length of its foot
are computed; frames within `corridor_m` are returned **ordered by arc-length**.
A drawn route re-orders frames, so `visloc_traj.py` uses uniform `--frame-dt`
spacing for any `--frame-subset` run (recorded timestamps are no longer monotone).
