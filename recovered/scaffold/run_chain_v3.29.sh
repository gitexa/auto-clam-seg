#!/usr/bin/env bash
# v3.29 / v3.32 / v3.33 chain.
# 1. build min2048 patch cache (CPU)
# 2. train v3.29 Stage 2 with min2048 (h=128)
# 3. cascade v3.31 = v3.8 + v3.29
# 4. train v3.32 Stage 1 n_hops=6
# 5. cascade v3.33 = v3.32 + v3.4b
# 6. cascade v3.34 = v3.32 + v3.29
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
V8_CKPT=$EXP/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt
V4B_CKPT=$EXP/gars_stage2_v3.4b_h128_min4096_20260430_090510/best_checkpoint.pt

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

# Step 1 — build min2048 cache
CACHE=/home/ubuntu/local_data/tls_patch_dataset_min2048.pt
if [ ! -s "$CACHE" ]; then
  log "Building min2048 patch cache"
  $PY -u tls_patch_dataset.py --min_tls_pixels 2048 --cache_path $CACHE
  log "Cache built"
fi

# Step 2 — Stage 2 v3.29 with min2048
log "Train v3.29 Stage 2 (h=128, min_tls_pixels=2048)"
PYTHONUNBUFFERED=1 nohup $PY -u train_gars_stage2.py \
  model=univ2_decoder_h128 label=v3.29_h128_min2048 \
  cache_path=$CACHE \
  train.class_weights=[1.0,5.0,3.0] train.gc_dice_weight=1.0 \
  > /tmp/v3.29.log 2>&1 &
sleep 60
V29_DIR=$(ls -1dt $EXP/gars_stage2_v3.29_h128_min2048_* 2>/dev/null | head -1)
log "v3.29 dir = $V29_DIR"
until [ -s $V29_DIR/best_checkpoint.pt ]; do sleep 30; done
until ! pgrep -f "python.*label=v3.29_h128_min2048" >/dev/null 2>&1; do sleep 30; done
log "v3.29 done"

# Step 3 — cascade v3.31
log "Cascade eval v3.31_s1v3.8_s2v3.29"
$PY -u eval_gars_cascade.py \
  stage1=$V8_CKPT stage2=$V29_DIR/best_checkpoint.pt \
  label=v3.31_s1v3.8_s2v3.29 \
  thresholds=[0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1

# Step 4 — Stage 1 v3.32 n_hops=6
log "Train v3.32 Stage 1 (n_hops=6)"
PYTHONUNBUFFERED=1 nohup $PY -u train_gars_stage1.py \
  model=gatv2_6hop label=v3.32_gatv2_6hop \
  > /tmp/v3.32.log 2>&1 &
sleep 60
V32_DIR=$(ls -1dt $EXP/gars_stage1_v3.32_gatv2_6hop_* 2>/dev/null | head -1)
log "v3.32 dir = $V32_DIR"
until [ -s $V32_DIR/best_checkpoint.pt ]; do sleep 30; done
until ! pgrep -f "python.*label=v3.32_gatv2_6hop" >/dev/null 2>&1; do sleep 30; done
log "v3.32 done"

# Step 5 — cascade v3.33
log "Cascade eval v3.33_s1v3.32_s2v3.4b"
$PY -u eval_gars_cascade.py \
  stage1=$V32_DIR/best_checkpoint.pt stage2=$V4B_CKPT \
  label=v3.33_s1v3.32_s2v3.4b \
  thresholds=[0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1

# Step 6 — cascade v3.34 (v3.32 + v3.29 if v3.29 was promising)
log "Cascade eval v3.34_s1v3.32_s2v3.29"
$PY -u eval_gars_cascade.py \
  stage1=$V32_DIR/best_checkpoint.pt stage2=$V29_DIR/best_checkpoint.pt \
  label=v3.34_s1v3.32_s2v3.29 \
  thresholds=[0.4,0.5] \
  min_component_size=2 closing_iters=1

log "v3.29/v3.32/v3.33/v3.34 chain done."
