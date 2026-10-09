#!/usr/bin/env bash
# Snapshot of the UAV-VisLoc zero-shot evaluation: what has been downloaded,
# which tile sizes have been prepared, and which evaluations have finished.
# Run with `watch -n 20 scripts/visloc_progress.sh` for a live view.
cd "$(dirname "$0")/.." || exit 1

echo "=== downloads (data/UAVVisLoc) ==============================="
for r in 01 02 03 04 05 06 07 08 09 10 11; do
  d=data/UAVVisLoc/$r
  [ -d "$d" ] || continue
  want=$(( $(wc -l < "$d/$r.csv" 2>/dev/null || echo 1) - 1 ))
  have=$(ls "$d/drone" 2>/dev/null | wc -l)
  tif=$(ls -la "$d"/satellite*.tif 2>/dev/null | awk '{printf "%.0f MB", $5/1048576}')
  [ "$have" = "0" ] && [ -z "$tif" ] && continue
  printf "  region %s   drone %4d/%-4d   mosaic %s\n" "$r" "$have" "$want" "${tif:---}"
done

echo
echo "=== prepared reference tile sets (data/visloc_avl) ============"
for d in data/visloc_avl/*/; do
  [ -f "$d/references.csv" ] || continue
  printf "  %-28s %5d tiles\n" "$(basename "$d")" "$(( $(wc -l < "$d/references.csv") - 1 ))"
done

echo
echo "=== finished evaluations (artifacts/visloc) ==================="
.venv/bin/python - <<'PY' 2>/dev/null
import json, glob, os
rows = []
for f in sorted(glob.glob('artifacts/visloc/*_summary.json')):
    d = json.load(open(f))
    rows.append((
        d['tag'], d['n_refs'], d['n_queries'],
        d['top1_error_m']['median'], d['top1_error_m']['p95'],
        100 * d['recall_top1_within_m']['100'],
        100 * d['recall_topk_within_m']['100'],
        100 * d.get('chance_baseline', {}).get('random_tile_within_m', {}).get('100', float('nan')),
    ))
if not rows:
    print('  (none yet)')
else:
    print(f"  {'run':<22}{'refs':>6}{'queries':>8}{'top1 med':>10}{'top1 p95':>10}"
          f"{'top1<100m':>11}{'top5<100m':>11}{'chance':>9}")
    for r in rows:
        print(f"  {r[0]:<22}{r[1]:>6}{r[2]:>8}{r[3]:>9.0f}m{r[4]:>9.0f}m"
              f"{r[5]:>10.1f}%{r[6]:>10.1f}%{r[7]:>8.1f}%")
PY

echo
echo "=== running now =============================================="
ps -eo etime,cmd | grep -E "visloc_(prepare|eval)|snapshot_download|satellite[0-9]+\.tif" | grep -v grep |
  sed 's/^/  /' | cut -c1-150
echo
echo "  live log:  tail -f $(ls -t /tmp/claude-1000/*/*/tasks/*.output 2>/dev/null | head -1)"
