#!/usr/bin/env bash
# Zero-shot DenseUAV-ViT evaluation on UAV-VisLoc, swept over reference tile ground size.
#
# The UAV-VisLoc drone frames carry no EXIF intrinsics, so the ground footprint of a
# frame cannot be computed and the correct reference tile size is unknown. Sweeping it
# both removes that unknown (report the best tile size) and measures scale sensitivity,
# which is itself a generalisation axis.
#
# Usage: scripts/visloc_sweep.sh <region> [tile sizes...]
set -euo pipefail

REGION="${1:?usage: visloc_sweep.sh <region> [tile_m ...]}"
shift || true
TILES=("$@")
if [ ${#TILES[@]} -eq 0 ]; then
  TILES=(150 300 600 1000)
fi

PY=.venv/bin/python
MAXQ="${MAXQ:-150}"
STRIDE="${STRIDE:-1}"

for T in "${TILES[@]}"; do
  OUT="data/visloc_avl/r${REGION}_t${T}"
  if [ ! -f "$OUT/references.csv" ]; then
    $PY scripts/visloc_prepare.py --region "$REGION" --tile-m "$T"
  fi
  $PY scripts/visloc_eval.py \
    --refs "$OUT/references.csv" \
    --queries "$OUT/queries.csv" \
    --rotations 4 \
    --query-crop square \
    --query-stride "$STRIDE" \
    --max-queries "$MAXQ" \
    --tag "r${REGION}_t${T}" \
    --ref-cache "artifacts/visloc/cache_r${REGION}_t${T}.npy"
done
