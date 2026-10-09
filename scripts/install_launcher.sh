#!/usr/bin/env bash
# Install a double-clickable launcher for the AVL Model Benchmark Console:
# one entry in the applications menu, one icon on the desktop.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
NAME="avl-model-bench"

mkdir -p "$APPS" "$ICONS"
install -m 644 "$ROOT/packaging/$NAME.svg" "$ICONS/$NAME.svg"

write_entry() {
  cat > "$1" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=AVL Model Benchmark Console
GenericName=Visual localization benchmark
Comment=Compare encoders and inspect top-5 retrievals with their coordinates
Exec=$ROOT/scripts/run-model-bench.sh
Path=$ROOT
Icon=$ICONS/$NAME.svg
Terminal=false
Categories=Science;
Keywords=AVL;localization;UAV;benchmark;
StartupNotify=true
EOF
  chmod +x "$1"
}

write_entry "$APPS/$NAME.desktop"
command -v desktop-file-validate >/dev/null 2>&1 && desktop-file-validate "$APPS/$NAME.desktop"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" || true

if [[ -d "$DESKTOP_DIR" ]]; then
  write_entry "$DESKTOP_DIR/$NAME.desktop"
  # GNOME refuses to launch a desktop file it has not been told to trust
  command -v gio >/dev/null 2>&1 && gio set "$DESKTOP_DIR/$NAME.desktop" metadata::trusted true || true
fi

echo "Installed:"
echo "  menu entry : $APPS/$NAME.desktop"
[[ -d "$DESKTOP_DIR" ]] && echo "  desktop    : $DESKTOP_DIR/$NAME.desktop"
echo "  icon       : $ICONS/$NAME.svg"
echo "Search for \"AVL Model Benchmark Console\" in Activities, or double-click the desktop icon."
