#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
pandoc "$ROOT/docs/AVL_Code_Guide.md" \
  -o "$ROOT/docs/AVL_Code_Guide.pdf" \
  --pdf-engine=xelatex \
  --include-in-header="$ROOT/docs/pandoc-header.tex" \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono"
echo "Wrote $ROOT/docs/AVL_Code_Guide.pdf"
