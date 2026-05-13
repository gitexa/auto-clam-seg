#!/usr/bin/env bash
# Strategy 2 round 1 eval: wait for v3.46 Stage 2 training (EM round 1 on
# v3.9 Stage 1 selections) to finish, then run cascade eval with v3.9 + v3.46.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments

V9_CKPT=$EXP/gars_stage1_v3.9_aux_finetune_fold0_20260512_224433/best_checkpoint.pt
V46_DIR=$(ls -1dt $EXP/gars_region_v3.46_em_round1_on_v3.9_* 2>/dev/null | head -1)

log() { echo "[v3.46 eval $(date +%H:%M:%S)] $*"; }
log "Waiting for v3.46 training to produce best_checkpoint.pt"
log "  v3.9 stage1 = $V9_CKPT"
log "  v3.46 dir   = $V46_DIR"

until [ -s "$V46_DIR/best_checkpoint.pt" ] && ! pgrep -f "label=v3.46_em_round1_on_v3.9" >/dev/null 2>&1; do
  sleep 60
  # refresh dir in case timestamp shifted
  V46_DIR=$(ls -1dt $EXP/gars_region_v3.46_em_round1_on_v3.9_* 2>/dev/null | head -1)
done
log "v3.46 training done. ckpt: $V46_DIR/best_checkpoint.pt"

# Smoke first (2 slides)
log "v3.46 smoke (2 slides)"
PYTHONUNBUFFERED=1 $PY -u eval_gars_cascade.py \
  stage1=$V9_CKPT \
  region_mode=true stage2_region=$V46_DIR/best_checkpoint.pt \
  thresholds=[0.5] limit_slides=2 \
  min_component_size=2 closing_iters=1 \
  label=v3.46_em_r1_smoke wandb.mode=disabled

# Full fold-0 eval at production post-proc; sweep thresholds
log "v3.46 full fold-0 eval"
PYTHONUNBUFFERED=1 $PY -u eval_gars_cascade.py \
  stage1=$V9_CKPT \
  region_mode=true stage2_region=$V46_DIR/best_checkpoint.pt \
  thresholds=[0.3,0.4,0.5,0.6,0.7] \
  min_component_size=2 closing_iters=1 \
  label=v3.46_em_r1_fold0_eval

log "v3.46 eval chain done"
