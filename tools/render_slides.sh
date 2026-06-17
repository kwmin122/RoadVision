#!/usr/bin/env bash
# RoadVision — Render all 11 slides to PNG via headless Chrome
set -euo pipefail

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SLIDES_DIR="$(cd "$(dirname "$0")/../presentation/slides" && pwd)"
OUT_DIR="$(cd "$(dirname "$0")/../presentation/render" && pwd)"

mkdir -p "$OUT_DIR"

for i in $(seq -w 1 11); do
  SRC="$SLIDES_DIR/slide-${i}.html"
  OUT="$OUT_DIR/slide${i}.png"
  echo "Rendering slide ${i} → $OUT"
  "$CHROME" \
    --headless=new \
    --disable-gpu \
    --hide-scrollbars \
    --force-device-scale-factor=1 \
    --screenshot="$OUT" \
    --window-size=1280,720 \
    "file://${SRC}" 2>/dev/null
done

echo ""
echo "Done. Rendered PNGs in: $OUT_DIR"
ls -lh "$OUT_DIR"/*.png
