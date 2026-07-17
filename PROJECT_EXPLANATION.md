# Project Explanation: Image-based Localization (GPS-Denied)

## Overview

This project implements Absolute Visual Localization (AVL) for UAVs when GPS is unavailable or unreliable. Instead of depending on live GPS, the system estimates the UAV position from an image by comparing it with a database of geo-tagged reference images.

The basic idea is:

1. Build a searchable visual map from reference images that already have latitude and longitude.
2. Encode a new UAV camera image into a neural visual descriptor.
3. Search for the most visually similar reference images.
4. Fuse the coordinates of the best matches into one estimated position.

The result is an estimated latitude, longitude, optional altitude, confidence score, and a ranked list of matching reference images.

## Problem This Project Solves

UAVs often need to navigate in environments where GPS is denied, jammed, spoofed, weak, or unavailable. This project provides a vision-based alternative: if the UAV camera sees terrain, buildings, roads, fields, or other recognizable visual patterns, the system can localize the image against a prebuilt geo-tagged image database.

This is useful for:

- UAV navigation in GPS-denied areas
- Search and rescue mapping
- Drone inspection workflows
- Offline visual map testing
- Evaluating visual place recognition models
- Comparing FAISS indexing methods for localization latency and accuracy

## High-level Pipeline

```text
Geo-tagged reference images
        |
        v
Visual encoder model
        |
        v
L2-normalized image descriptors
        |
        v
FAISS nearest-neighbor index
        |
        v
Query UAV image
        |
        v
Top-K visual matches
        |
        v
Weighted geographic fusion
        |
        v
Estimated lat/lon/alt
```

## Main Workflows

### 1. Build an Offline Reference Index

The offline step reads a CSV file of geo-tagged reference images, encodes each image with a visual place recognition model, and stores the descriptors in a FAISS index.

Example:

```bash
python scripts/build_index.py \
  --metadata examples/references.csv \
  --base-dir . \
  --output artifacts/demo_index
```

This creates files such as:

- `index.faiss`: the FAISS search index
- `metadata.json`: saved model/index configuration and reference records
- `build_report.json`: offline timing and index-size metrics

### 2. Localize a Query Image

The online step loads the saved index, encodes a query UAV image, searches for the closest reference descriptors, and estimates a geographic pose.

Example:

```bash
python scripts/localize.py \
  --index artifacts/demo_index \
  --query examples/queries/query.jpg \
  --top-k 5
```

The output includes:

- Estimated latitude and longitude
- Optional altitude
- Confidence score
- Best visual match
- Ranked Top-K matches

### 3. Benchmark Models and Index Types

The project includes KPI benchmarking for comparing model speed, FAISS search speed, index size, recall, and localization error.

Example:

```bash
python scripts/benchmark_kpis.py \
  --metadata examples/references.csv \
  --base-dir . \
  --query examples/queries/query.jpg \
  --model-config denseuav-vit:512 \
  --index-types hnsw flat \
  --top-k 5 \
  --repeats 3 \
  --output-json artifacts/kpis.json \
  --output-csv artifacts/kpis.csv
```

Important KPI groups:

- Offline encoding time for reference images
- FAISS index build time
- Serialized index size
- Query image encoding latency
- Search latency
- Encode-plus-search online latency
- Recall@K when expected reference IDs are available
- Top-1 and fused localization error when ground-truth coordinates are available

### 4. Use the GUI Console

The repository also includes a lightweight PySide6 desktop app called the AVL Mission Console.

Run it with:

```bash
python scripts/run_desktop_app.py
```

The GUI supports:

- Building the offline reference index
- Running repeated online localization queries
- Keeping the model and index loaded in a persistent worker process
- Viewing estimated coordinates and Top-K reference matches
- Running batch KPI evaluation
- Loading previously generated KPI JSON reports

## Input Data Format

### Reference CSV

The reference CSV describes the searchable geo-tagged image database.

Required columns:

- `image_path`
- `latitude`
- `longitude`

Optional columns:

- `altitude_m`
- `heading_deg`
- `image_id`

Example:

```csv
image_path,latitude,longitude,altitude_m,heading_deg,image_id
data/refs/img001.jpg,48.8584,2.2945,50.0,180.0,img001
```

### Query CSV for Benchmarking

Batch benchmarking can use a query CSV.

Required column:

- `image_path`

Optional columns:

- `latitude`
- `longitude`
- `expected_image_id`

Example:

```csv
image_path,latitude,longitude,expected_image_id
examples/queries/query.jpg,48.85854,2.29464,ref_04
```

`expected_image_id` is used to compute Recall@1, Recall@5, and Recall@10.

## Core Components

| File | Purpose |
|------|---------|
| `avl/config.py` | Stores runtime configuration such as model name, descriptor size, device, FAISS index type, Top-K, and query rotations. |
| `avl/metadata.py` | Loads and validates reference CSV files and converts rows into `ReferenceRecord` objects. |
| `avl/encoder.py` | Loads the selected neural model and converts images into normalized visual descriptors. |
| `avl/index.py` | Builds, searches, saves, and loads FAISS indexes. Supports `flat`, `hnsw`, and `ivfpq`. |
| `avl/localizer.py` | Connects the full localization pipeline: encode query, search index, rank matches, fuse coordinates, and return a result. |
| `avl/geo.py` | Contains haversine distance and weighted geographic fusion logic. |
| `avl/models/` | Contains model loaders for DenseUAV ViT, MixVPR, and CosPlace. |
| `avl/light_gui_app.py` | Implements the desktop GUI mission console. |
| `scripts/build_index.py` | CLI for building an offline reference index. |
| `scripts/localize.py` | CLI for localizing one query image. |
| `scripts/query_index.py` | Online-only localization script with timing output against an existing index. |
| `scripts/benchmark_kpis.py` | CLI for model/index benchmarking and quality metrics. |
| `scripts/download_denseuav.py` | Downloads the DenseUAV dataset from Hugging Face. |
| `scripts/prepare_denseuav.py` | Converts DenseUAV metadata into this project's CSV format. |
| `scripts/create_examples.py` | Creates synthetic example images for a smoke test. |
| `scripts/online_worker.py` | Persistent online engine used by the GUI to avoid reloading the model for every query. |

## Models

The system supports three visual descriptor backends.

| Model | Descriptor Dimension | Notes |
|-------|----------------------|-------|
| `denseuav-vit` | 512 | Default model. Designed for UAV-to-satellite localization. |
| `mixvpr` | 4096 or 512 | Strong visual place recognition baseline. |
| `cosplace` | 2048 | Alternative place-recognition model loaded through PyTorch Hub style tooling. |

All descriptors are L2-normalized before search. Because the FAISS indexes use inner product similarity, normalized vectors make the search behave like cosine similarity.

## FAISS Index Types

| Index Type | Best Use |
|------------|----------|
| `flat` | Exact search. Good for small datasets and debugging. |
| `hnsw` | Fast approximate search with strong recall. Good default for medium to large galleries. |
| `ivfpq` | Compressed approximate search. Useful for very large maps or memory-constrained deployments. |

## How Localization Is Estimated

When a query image is localized:

1. The query image is encoded into one or more descriptors.
2. If `query_rotations=4`, the query is also tested at 0, 90, 180, and 270 degrees.
3. FAISS returns the most similar reference records.
4. Duplicate location IDs are collapsed so one location does not dominate the result.
5. The top matches are converted into latitude and longitude arrays.
6. The top five match scores are used as weights.
7. Coordinates are averaged on the unit sphere to avoid longitude wrap problems.
8. A confidence score is computed from the best match score and geographic spread of the fused matches.

This means the final position is not just the coordinate of the single best image. It is a weighted fusion of the strongest nearby visual matches.

## DenseUAV Dataset Support

The project includes helper scripts for DenseUAV, a UAV localization dataset with drone-view and satellite-view imagery.

Download:

```bash
python scripts/download_denseuav.py \
  --output-dir data/DenseUAV \
  --download-dir data/downloads
```

Prepare AVL-compatible CSV files:

```bash
python scripts/prepare_denseuav.py \
  --dense-root data/DenseUAV \
  --output-dir data/denseuav_avl \
  --reference-split gallery \
  --reference-view satellite \
  --query-view drone
```

Typical DenseUAV usage is to use satellite gallery images as the reference database and drone test images as queries.

## Installation

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The main dependencies are:

- PyTorch and TorchVision for neural inference
- FAISS for vector search
- NumPy and pandas for data handling
- Pillow and OpenCV for image processing
- PySide6 for the GUI
- Hugging Face Hub and timm for model/dataset support

## Quick Smoke Test

Generate synthetic data:

```bash
python scripts/create_examples.py
```

Build the demo index:

```bash
python scripts/build_index.py \
  --metadata examples/references.csv \
  --base-dir . \
  --output artifacts/demo_index
```

Localize the demo query:

```bash
python scripts/localize.py \
  --index artifacts/demo_index \
  --query examples/queries/query.jpg
```

## What This Project Does Not Include

This repository focuses on visual localization and benchmarking. It does not currently provide:

- Full UAV flight control
- Multi-frame temporal filtering
- SLAM or visual odometry
- Real-time camera capture integration
- Geographic tile pre-filtering at runtime
- Mission planning or autopilot integration

Those pieces could be added around this project. The current code is mainly the localization engine, data preparation tools, benchmark tooling, and desktop test console.

## Practical Notes

- First use of some models may download weights into a local cache.
- GPU inference is preferred, but the code falls back to CPU if CUDA is unavailable.
- Reference and query imagery should have similar viewpoint and appearance for best results.
- Dense, well-distributed reference coverage improves localization accuracy.
- The `score_threshold` prevents low-confidence matches from being accepted as valid localizations.
- For large galleries, index type and descriptor dimension have a major impact on latency and memory use.

## One-sentence Summary

This project builds a searchable visual map from geo-tagged images and uses neural image retrieval plus FAISS search to estimate a UAV image's GPS-like position without relying on GPS.
