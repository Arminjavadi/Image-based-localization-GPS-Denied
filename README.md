# Image-based Localization (GPS-Denied)

Absolute Visual Localization (AVL) for UAVs using geo-tagged reference imagery, state-of-the-art visual place recognition (VPR), and FAISS approximate nearest-neighbor search.

## Pipeline

```
Geo-tagged reference images
        │
        ▼
MixVPR / CosPlace encoder  ──►  L2-normalized descriptors
        │
        ▼
FAISS index (HNSW or IVF-PQ hashing)
        │
        ▼
Query UAV image  ──►  top-K retrieval  ──►  weighted geo fusion  ──►  lat/lon/alt
```

## Features

- **MixVPR (default)** — SOTA holistic VPR aggregator (WACV 2023), 4096-dim descriptors
- **CosPlace** — strong alternative via PyTorch Hub (`ResNet101`, 2048-dim)
- **FAISS hashing**
  - `hnsw` — best recall/latency for medium databases (default)
  - `ivfpq` — product-quantized compressed index for large-scale maps
  - `flat` — exact baseline
- **Geo fusion** — sphere-aware weighted average of top-K match coordinates

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
