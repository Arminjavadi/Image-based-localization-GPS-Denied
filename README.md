# Image-based Localization (GPS-Denied)

Absolute Visual Localization (AVL) for UAVs using geo-tagged reference imagery, state-of-the-art visual place recognition (VPR), and FAISS approximate nearest-neighbor search.

## Pipeline

```
Geo-tagged reference images
        │
        ▼
 DenseUAV ViT / MixVPR / CosPlace encoder  ──►  L2-normalized descriptors
        │
        ▼
FAISS index (HNSW or IVF-PQ hashing)
        │
        ▼
Query UAV image  ──►  top-K retrieval  ──►  weighted geo fusion  ──►  lat/lon/alt
```

## Features

- **DenseUAV ViT (default)** — 512-dim cross-view descriptor trained for UAV-to-satellite retrieval
- **MixVPR** — strong general-purpose VPR baseline, 4096-dim descriptors
- **CosPlace** — strong alternative via PyTorch Hub (`ResNet101`, 2048-dim)
- **FAISS hashing**
  - `flat` — exact search and the default for DenseUAV-sized galleries
  - `hnsw` — strong recall/latency for much larger databases
  - `ivfpq` — product-quantized compressed index for large-scale maps
- **Geo fusion** — sphere-aware weighted average of the five strongest unique locations

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

## Production usage

### 1. Build index from your geo-tagged map

```bash
python scripts/build_index.py \
  --metadata data/references.csv \
  --base-dir data/ \
  --output artifacts/uav_map_index \
  --model mixvpr \
  --descriptor-dim 4096 \
  --index-type hnsw
```

For very large databases (millions of tiles), use IVF-PQ hashing:

```bash
python scripts/build_index.py \
  --metadata data/references.csv \
  --base-dir data/ \
  --output artifacts/uav_map_index \
  --index-type ivfpq
```

### 2. Localize a UAV frame

```bash
python scripts/localize.py \
  --index artifacts/uav_map_index \
  --query path/to/uav_frame.jpg \
  --top-k 5 \
  --json-out outputs/localization.json
```

### 3. Python API

```python
from pathlib import Path
from avl import AVLLocalizer
from avl.config import AVLConfig

config = AVLConfig(model="mixvpr", descriptor_dim=4096, index_type="hnsw")
localizer = AVLLocalizer(config)

localizer.build_index(
    metadata_csv=Path("data/references.csv"),
    output_dir=Path("artifacts/index"),
    base_dir=Path("data"),
)

localizer.load_index(Path("artifacts/index"))
result = localizer.localize(Path("query.jpg"))
print(result.pose.latitude, result.pose.longitude, result.confidence)
```

## Model choices

| Model | Descriptor dim | Best for |
|-------|----------------|----------|
| `mixvpr` (default) | 4096 | Highest VPR accuracy |
| `mixvpr` | 512 | Faster indexing, lower memory |
| `cosplace` | 2048 | Alternative backbone, PyTorch Hub |

```bash
python scripts/build_index.py --model cosplace --descriptor-dim 2048 --cosplace-backbone ResNet101 ...
```

## Index tuning

| Index | When to use |
|-------|-------------|
| `hnsw` | < 1M references, best accuracy/speed trade-off |
| `ivfpq` | Large maps, memory-constrained deployment |
| `flat` | Small maps, debugging |

## UAV tips

1. Restrict search to a geographic tile when a coarse prior exists (pre-filter your CSV before indexing).
2. Use multi-frame temporal filtering in your flight stack (not included here).
3. Match reference viewpoint: nadir UAV frames work best against nadir/satellite references.
4. First run downloads MixVPR weights (~150 MB) to `~/.cache/avl/mixvpr/`.

## Project layout

```
avl/
  encoder.py      # MixVPR / CosPlace batch encoder
  index.py        # FAISS HNSW / IVF-PQ index
  localizer.py    # End-to-end AVL
  geo.py          # Haversine + weighted fusion
scripts/
  build_index.py
  localize.py
  create_examples.py
```

## References

- [MixVPR](https://arxiv.org/abs/2303.02190) — Amar Ali-bey et al., WACV 2023
- [CosPlace](https://arxiv.org/abs/2204.02287) — Berton et al., CVPR 2022
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
