---
title: "Absolute Visual Localization"
subtitle: "Code Architecture & Developer Guide"
author: "Image-based Localization (GPS-Denied)"
date: "June 2026"
toc: false
toc-depth: 2
numbersections: true
colorlinks: true
linkcolor: avlblue
urlcolor: avlteal
geometry: margin=2.5cm
fontsize: 11pt
---

\begin{titlepage}
\centering
\vspace*{2cm}
{\Huge\bfseries\textcolor{avlblue}{Absolute Visual Localization}\par}
\vspace{0.6cm}
{\Large\textcolor{avlteal}{Code Architecture \& Developer Guide}\par}
\vspace{1.2cm}
{\large Image-based Localization (GPS-Denied)\par}
\vspace{0.4cm}
{\large June 2026\par}
\vfill
\begin{tcolorbox}[colback=avlgray, colframe=avlblue, width=0.85\textwidth, arc=3mm]
\centering
\small
Encode images $\rightarrow$ FAISS search $\rightarrow$ GPS fusion\\
State-of-the-art MixVPR + approximate nearest-neighbor hashing
\end{tcolorbox}
\vspace{1.5cm}
\end{titlepage}
\tableofcontents
\newpage

# Introduction

This document explains the **Absolute Visual Localization (AVL)** codebase: a GPS-denied UAV localization system that matches live camera images against a database of **geo-tagged reference images**.

\begin{infobox}{What problem does this solve?}
When GPS is unavailable or unreliable, a UAV can still estimate its global position by asking: \emph{``Which known place does my camera see right now?''} Each reference image carries a known latitude/longitude. The closest visual match provides the location estimate.
\end{infobox}

**Core pipeline in one sentence:**

> Encode images into vectors → search similar vectors with FAISS → fuse GPS coordinates of top matches.

# System Architecture

## High-Level Overview

The system has two phases:

| Phase | When | Input | Output |
|-------|------|-------|--------|
| **Index build** (offline) | Once per map | Geo-tagged reference CSV + images | `index.faiss` + `metadata.json` |
| **Localization** (online) | Per UAV frame | Query image | lat, lon, altitude, confidence |

## Data Flow Diagram

```{=latex}
\begin{center}
\begin{tikzpicture}[
  node distance=0.55cm and 0.9cm,
  box/.style={draw=avlborder, fill=avlgray, rounded corners=3pt,
              minimum width=2.6cm, minimum height=0.85cm, align=center, font=\small},
  arrow/.style={-{Stealth[length=2.2mm]}, thick, color=avlblue}
]
  \node[box] (csv) {references.csv};
  \node[box, right=of csv] (meta) {metadata.py};
  \node[box, right=of meta] (enc) {VPREncoder};
  \node[box, below=of enc] (idx) {HashIndex (FAISS)};
  \node[box, left=of idx] (save) {index.faiss\\metadata.json};

  \node[box, below=1.4cm of save] (query) {Query image};
  \node[box, right=of query] (enc2) {VPREncoder};
  \node[box, right=of enc2] (search) {FAISS search};
  \node[box, right=of search] (geo) {geo.py fusion};
  \node[box, right=of geo] (pose) {GeoPose};

  \draw[arrow] (csv) -- (meta);
  \draw[arrow] (meta) -- (enc);
  \draw[arrow] (enc) -- (idx);
  \draw[arrow] (idx) -- (save);

  \draw[arrow] (query) -- (enc2);
  \draw[arrow] (enc2) -- (search);
  \draw[arrow] (save.south) |- (search.south);
  \draw[arrow] (search) -- (geo);
  \draw[arrow] (geo) -- (pose);
\end{tikzpicture}
\end{center}
```

## Project Structure

```
Image-based-localization-GPS-Denied/
├── avl/                      # Core library
│   ├── config.py             # All tunable parameters
│   ├── encoder.py            # Neural image → descriptor
│   ├── index.py              # FAISS approximate search
│   ├── geo.py                # GPS math & fusion
│   ├── localizer.py          # Main orchestrator
│   ├── metadata.py           # CSV loading
│   └── models/
│       ├── mixvpr.py         # MixVPR network (default)
│       └── cosplace.py       # CosPlace alternative
├── scripts/
│   ├── build_index.py        # CLI: build FAISS index
│   ├── localize.py           # CLI: localize query image
│   └── create_examples.py    # Synthetic demo data
└── examples/
    └── references.csv        # Sample metadata format
```

# Module Reference

## `AVLLocalizer` — The Main Entry Point

**File:** \filepath{avl/localizer.py}

This class orchestrates the entire pipeline. It owns a `VPREncoder` and a `HashIndex`.

### Building an index (offline)

```python
def build_index(self, metadata_csv, output_dir, base_dir=None):
    records = load_reference_metadata(metadata_csv, base_dir=base_dir)
    image_paths = [record.image_path for record in records]
    descriptors = self.encoder.encode_paths(image_paths)
    self.index.build(descriptors, records)
    self.index.save(output_dir)
```

**Step-by-step:**

1. Parse the CSV into `ReferenceRecord` objects (path + GPS).
2. Encode every reference image into a descriptor matrix of shape `(N, D)`.
3. Insert descriptors into FAISS and attach GPS metadata.
4. Persist to disk for later reuse.

### Localizing a query (online)

```python
def localize(self, query_image, top_k=None):
    descriptor = self.encoder.encode_image(query_image)
    search = self.index.search(descriptor, top_k=top_k)
    # ... build Match list from search results ...
    pose = weighted_geo_fusion(latitudes, longitudes, weights)
    confidence = best.score * exp(-spread_m / 500)
    return LocalizationResult(...)
```

**Step-by-step:**

1. Encode the UAV query frame.
2. Retrieve top-$K$ most similar reference descriptors.
3. Reject if best score $<$ `score_threshold` (default 0.35).
4. Fuse GPS of top-$K$ matches (weighted by similarity).
5. Compute confidence from match quality and geographic agreement.

\begin{infobox}{Key data classes}
\textbf{Match} — one retrieved reference: rank, cosine score, GPS record.\\
\textbf{LocalizationResult} — final pose, confidence, best match, all matches, query descriptor.
\end{infobox}

## `VPREncoder` — Image Fingerprints

**File:** \filepath{avl/encoder.py}

Converts RGB images into fixed-size **descriptor vectors** using a Visual Place Recognition (VPR) neural network.

### Supported models

| Model | Default dim | Input size | Source |
|-------|-------------|------------|--------|
| **MixVPR** | 4096 | 320×320 | Pretrained weights via `gdown` |
| **CosPlace** | 2048 | 322×322 | PyTorch Hub (`gmberton/CosPlace`) |

### Encoding pipeline

```python
# For each image batch:
batch = transform(image)          # resize + normalize
outputs = model(batch)            # neural network forward pass
outputs = F.normalize(outputs)    # L2 normalize → unit vector
```

After L2 normalization, the **dot product** between two descriptors equals **cosine similarity** (range roughly $-1$ to $1$; higher = more similar places).

\begin{warnbox}{Why normalization matters}
FAISS is configured with \texttt{METRIC\_INNER\_PRODUCT}. This only equals cosine similarity when all vectors are unit-length. Both the encoder and index normalize vectors before search.
\end{warnbox}

### MixVPR internals

**File:** \filepath{avl/models/mixvpr.py}

MixVPR (WACV 2023) uses two components:

1. **ResNet50 backbone** (layers 1–3 only) — extracts spatial feature maps.
2. **MixVPR aggregator** — mixes feature-map relationships with MLP blocks, then projects to a compact global descriptor.

```
Image (320×320)
    → ResNet50 (conv1 … layer3)
    → Feature maps (1024 × 20 × 20)
    → FeatureMixer layers (×4)
    → Channel + row projections
    → L2-normalized descriptor (4096-dim)
```

Weights are downloaded once to `~/.cache/avl/mixvpr/`.

## `HashIndex` — FAISS Similarity Search

**File:** \filepath{avl/index.py}

FAISS (**F**acebook **AI** **S**imilarity **S**earch) finds the nearest descriptor vectors without comparing against every reference.

### Index types

| Type | Class | Speed | Accuracy | Use case |
|------|-------|-------|----------|----------|
| `flat` | `IndexFlatIP` | Slowest | Exact | Debugging, small maps |
| `hnsw` | `IndexHNSWFlat` | Fast | Near-exact | **Default** — up to ~1M images |
| `ivfpq` | `IndexIVFPQ` | Very fast | Approximate | Millions of tiles, low memory |

**HNSW** builds a navigable graph over vectors. Search follows graph edges to quickly reach the nearest neighbors.

**IVF-PQ** clusters vectors (Inverted File) and stores compressed Product Quantization codes — this is the ``hashing'' approach for massive databases.

### Search

```python
faiss.normalize_L2(query)
scores, indices = index.search(query, top_k)
```

Returns the indices of the $K$ closest references and their similarity scores.

### Persistence

| File | Contents |
|------|----------|
| `index.faiss` | FAISS binary index (vectors + search structure) |
| `metadata.json` | Model config + GPS record per indexed image |

## `geo.py` — Geographic Math

**File:** \filepath{avl/geo.py}

### Haversine distance

Computes great-circle distance in meters between two GPS points on the Earth's surface. Used to measure how spread out the top-$K$ matches are (for confidence).

### Weighted geo fusion

Instead of trusting only the single best match, the system fuses top-$K$ GPS coordinates weighted by similarity score.

**Critical detail:** latitude and longitude are **not** averaged directly. They are converted to 3D unit-sphere coordinates $(x, y, z)$, weighted-averaged, then converted back:

```python
x = cos(lat) * cos(lon)
y = cos(lat) * sin(lon)
z = sin(lat)
# weighted mean of x, y, z → back to lat/lon
```

This avoids errors near the poles and the $\pm 180°$ longitude wrap.

### Confidence formula

$$\text{confidence} = \text{best\_score} \times e^{-\text{spread\_m} / 500}$$

- High best score → confident.
- Top matches agree geographically (low spread) → confident.
- Matches scattered far apart → confidence drops.

## `metadata.py` — Reference Database Format

**File:** \filepath{avl/metadata.py}

### Required CSV columns

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | string | Path to reference image |
| `latitude` | float | WGS84 latitude (degrees) |
| `longitude` | float | WGS84 longitude (degrees) |

### Optional columns

| Column | Description |
|--------|-------------|
| `altitude_m` | Altitude in meters |
| `heading_deg` | Camera heading |
| `image_id` | Custom identifier |

**Example:**

```csv
image_path,latitude,longitude,altitude_m,heading_deg,image_id
data/refs/tile001.jpg,48.8584,2.2945,50.0,180.0,tile001
```

## `AVLConfig` — Configuration

**File:** \filepath{avl/config.py}

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model` | `mixvpr` | VPR backbone (`mixvpr` or `cosplace`) |
| `descriptor_dim` | `4096` | Vector size |
| `index_type` | `hnsw` | FAISS index algorithm |
| `top_k` | `5` | Matches used for geo fusion |
| `score_threshold` | `0.35` | Minimum similarity to accept |
| `device` | `cuda` | PyTorch device (falls back to CPU) |
| `batch_size` | `16` | Encoding batch size |

# Command-Line Tools

## Build index

```bash
python scripts/build_index.py \
  --metadata data/references.csv \
  --base-dir data/ \
  --output artifacts/uav_map_index \
  --model mixvpr \
  --descriptor-dim 4096 \
  --index-type hnsw
```

## Localize query

```bash
python scripts/localize.py \
  --index artifacts/uav_map_index \
  --query path/to/uav_frame.jpg \
  --top-k 5 \
  --json-out outputs/result.json
```

**JSON output fields:**

| Field | Description |
|-------|-------------|
| `latitude`, `longitude` | Fused position estimate |
| `altitude_m` | Weighted altitude (if available) |
| `confidence` | Quality score (0–1 range) |
| `best_match` | Top-1 reference details |
| `matches` | Full top-$K$ list with scores |

# Python API Example

```python
from pathlib import Path
from avl import AVLLocalizer
from avl.config import AVLConfig

config = AVLConfig(model="mixvpr", descriptor_dim=4096, index_type="hnsw")
localizer = AVLLocalizer(config)

# Offline: build once
localizer.build_index(
    metadata_csv=Path("data/references.csv"),
    output_dir=Path("artifacts/index"),
    base_dir=Path("data"),
)

# Online: per frame
localizer.load_index(Path("artifacts/index"))
result = localizer.localize(Path("query.jpg"))

print(f"Position: {result.pose.latitude:.6f}, {result.pose.longitude:.6f}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Best match: {result.best_match.record.image_path}")
```

# Design Decisions & Limitations

## What this system does well

- Fast retrieval over large geo-tagged image databases.
- State-of-the-art VPR descriptors (MixVPR).
- Configurable FAISS indexes for speed vs. scale.
- Robust GPS fusion from multiple matches.
- Simple CSV-based reference format.

## Current limitations

\begin{warnbox}{Not included in this version}
\begin{itemize}[leftmargin=*]
  \item \textbf{No keypoint matching} — no SIFT/SuperPoint geometric verification.
  \item \textbf{No temporal filtering} — single-frame only; no video smoothing.
  \item \textbf{No GPS prior} — search is global; no bounding-box restriction.
  \item \textbf{Cross-view gap} — nadir UAV vs.\ oblique satellite may need specialized models.
\end{itemize}
\end{warnbox}

## Recommended extensions for production UAV use

1. **Geographic pre-filter** — restrict search to a tile around last known GPS.
2. **Multi-frame voting** — median or Kalman filter over consecutive frames.
3. **Local feature verification** — SuperPoint + LightGlue on top-$K$ candidates.
4. **Altitude aiding** — barometer + homography scale consistency check.

# Glossary

| Term | Definition |
|------|------------|
| **AVL** | Absolute Visual Localization — global lat/lon from images |
| **VPR** | Visual Place Recognition — matching places by appearance |
| **Descriptor** | Fixed-size numeric fingerprint of an image |
| **FAISS** | Library for fast approximate nearest-neighbor search |
| **HNSW** | Hierarchical Navigable Small World graph index |
| **IVF-PQ** | Inverted File + Product Quantization (compressed hashing) |
| **Cosine similarity** | Measure of vector similarity ($-1$ to $1$) |
| **Top-$K$ fusion** | Combining $K$ best matches into one GPS estimate |
| **Geo-tagged** | Image with known GPS coordinates attached |

# References

- MixVPR: Ali-bey et al., WACV 2023 — \url{https://arxiv.org/abs/2303.02190}
- CosPlace: Berton et al., CVPR 2022 — \url{https://arxiv.org/abs/2204.02287}
- FAISS: Johnson et al. — \url{https://github.com/facebookresearch/faiss}
