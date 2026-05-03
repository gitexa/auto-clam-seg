#!/usr/bin/env bash
# v3.27 chain: wait for h=256 Stage 2, then cascade with v3.8 Stage 1.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
V8_CKPT=$EXP/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

V27_DIR=$(ls -1dt $EXP/gars_stage2_v3.27_h256_min4096_* 2>/dev/null | head -1)
log "v3.27 dir = $V27_DIR"

log "Waiting for v3.27 Stage 2 best_checkpoint"
until [ -s $V27_DIR/best_checkpoint.pt ]; do sleep 30; done
log "Found $V27_DIR/best_checkpoint.pt"
log "Waiting for v3.27 process to exit"
until ! pgrep -f "python.*label=v3.27_h256_min4096" >/dev/null 2>&1; do sleep 30; done

log "Cascade eval: v3.30_s1v3.8_s2v3.27"
PYTHONUNBUFFERED=1 $PY -u eval_gars_cascade.py \
  stage1=$V8_CKPT stage2=$V27_DIR/best_checkpoint.pt \
  label=v3.30_s1v3.8_s2v3.27 \
  thresholds=[0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1

log "v3.27 chain done."
