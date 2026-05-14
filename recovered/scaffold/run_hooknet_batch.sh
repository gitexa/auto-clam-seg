#!/usr/bin/env bash
# Batch HookNet-TLS inference on fold-0 val slides.
#
# For each slide:
#   1. preprocessing docker -> pyramid TIF + tissue mask TIF (~3 min)
#   2. hooknet-tls docker -> prediction mask, heatmaps, filtered JSON (~5-10 min)
# Outputs land in /home/ubuntu/local_data/hooknet_out/<slide>/
#
# ~10-15 min/slide -> ~30h for 165 slides.
set -uo pipefail
OUT_BASE=/home/ubuntu/local_data/hooknet_out

# Get fold-0 val slide list (uses the same prepare_segmentation split)
SLIDE_LIST=/tmp/fold0_val_slides.txt
PY=/home/ubuntu/profile-clam/.venv/bin/python
$PY -c "
import sys
sys.path.insert(0, '/home/ubuntu/auto-clam-seg/recovered/scaffold')
sys.path.insert(0, '/home/ubuntu/profile-clam')
import prepare_segmentation as ps
from gncaf_dataset import slide_wsi_path
entries = ps.build_slide_entries()
folds, _ = ps.create_splits(entries, k_folds=5, seed=42)
for e in folds[0]:
    p = slide_wsi_path(e)
    if p:
        sid = e['slide_id'].split('.')[0]
        print(f'{sid}|{p}')
" > $SLIDE_LIST

N_TOTAL=$(wc -l < $SLIDE_LIST)
echo "[batch $(date +%H:%M)] $N_TOTAL slides queued"

i=0
while IFS='|' read -r SID WSI; do
  i=$((i+1))
  SLIDE_DIR=$(dirname "$WSI")
  SLIDE_FN=$(basename "$WSI")
  OUT=$OUT_BASE/$SID
  mkdir -p "$OUT"
  # Skip if already done
  if [ -f "$OUT/images/post-processed/${SID}_hooknettls_post_processed.xml" ]; then
    echo "[batch $(date +%H:%M)] [$i/$N_TOTAL] $SID SKIP (already done)"
    continue
  fi
  echo "[batch $(date +%H:%M)] [$i/$N_TOTAL] $SID START"

  # Preprocessing
  if [ ! -f "$OUT/${SID}.tif" ] || [ ! -f "$OUT/${SID}_mask.tif" ]; then
    sudo docker run --rm \
      -v "$SLIDE_DIR":/input:ro \
      -v "$OUT":/output \
      hooknet-preprocessing \
      "/input/${SLIDE_FN}" "/output/${SID}.tif" "/output/${SID}_mask.tif" \
      > "$OUT/preproc.log" 2>&1 || { echo "  preproc FAIL"; continue; }
  fi

  # HookNet inference
  sudo docker run --rm --gpus all \
    -v "$OUT":/output \
    hooknet-tls \
    python3 -m hooknettls \
      hooknettls.default.image_path=/output/${SID}.tif \
      hooknettls.default.mask_path=/output/${SID}_mask.tif \
      hooknettls.default.iterator.cpus=1 \
    > "$OUT/hooknet.log" 2>&1 || { echo "  hooknet FAIL"; continue; }

  # Cleanup big intermediates to save space
  rm -f "$OUT/${SID}.tif"  # the converted pyramid TIF (~800 MB)
  rm -f "$OUT/images/${SID}_hooknettls_heat1.tif" "$OUT/images/${SID}_hooknettls_heat2.tif"
  echo "[batch $(date +%H:%M)] [$i/$N_TOTAL] $SID DONE"
done < $SLIDE_LIST
echo "[batch $(date +%H:%M)] all done"
