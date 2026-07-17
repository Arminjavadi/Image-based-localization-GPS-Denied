#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv "$ROOT/.venv"
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$ROOT/requirements.txt"

rm -rf "$ROOT/build/AVL-Mission-Console" "$ROOT/dist/AVL-Mission-Console" "$ROOT/AVL-Mission-Console.spec"

"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --name AVL-Mission-Console \
  --onedir \
  --windowed \
  --exclude-module torch \
  --exclude-module torchvision \
  --exclude-module faiss \
  --exclude-module pandas \
  --exclude-module numpy \
  --exclude-module cv2 \
  --exclude-module timm \
  --exclude-module huggingface_hub \
  --exclude-module avl.encoder \
  --exclude-module avl.localizer \
  --exclude-module avl.models \
  --exclude-module avl.gui_app \
  "$ROOT/scripts/run_desktop_app.py"

echo "Built Ubuntu executable: $ROOT/dist/AVL-Mission-Console/AVL-Mission-Console"
