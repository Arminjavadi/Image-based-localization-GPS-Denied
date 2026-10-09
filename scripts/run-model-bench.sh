#!/usr/bin/env bash
# Clickable launcher for the AVL Model Benchmark Console.
#
# The console shells out to scripts/visloc_eval.py with sys.executable, so it has
# to run under the project's own interpreter (torch, faiss, timm) rather than a
# frozen bundle. This script finds that interpreter and hands over to it.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
LOG="$ROOT/artifacts/model_bench_launcher.log"
mkdir -p "$(dirname "$LOG")"

fail() {
  echo "$1" >>"$LOG"
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="AVL Model Benchmark Console" --width=420 --text="$1"
  fi
  exit 1
}

PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3 || true)"
[[ -n "$PYTHON" && -x "$PYTHON" ]] || fail "No Python interpreter found.\nExpected $ROOT/.venv/bin/python"

"$PYTHON" -c "import PySide6, pandas" 2>>"$LOG" \
  || fail "PySide6 and pandas are missing from $PYTHON.\nInstall them with:\n  $PYTHON -m pip install -r $ROOT/requirements.txt"

cd "$ROOT"
exec "$PYTHON" "$ROOT/scripts/run_model_bench.py" "$@" >>"$LOG" 2>&1
