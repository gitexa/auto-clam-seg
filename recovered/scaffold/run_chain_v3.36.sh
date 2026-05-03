#!/usr/bin/env bash
# Wait for v3.36 NeighborhoodPixelDecoder training, then run G3 (smoke)
# + G4 (full benchmark) cascade evals via the extended eval_gars_cascade.py.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
V8_CKPT=$EXP/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

V36_DIR=$(ls -1dt $EXP/gars_neighborhood_v3.36_full_* 2>/dev/null | head -1)
log "v3.36 dir = $V36_DIR"

log "Waiting for v3.36 training to finish"
until ! pgrep -f "python.*label=v3.36_full" >/dev/null 2>&1; do sleep 30; done
log "v3.36 training done"
if [ ! -s "$V36_DIR/best_checkpoint.pt" ]; then
  log "ERR: no checkpoint in $V36_DIR"; exit 1
fi
log "ckpt = $V36_DIR/best_checkpoint.pt"

# G3 — smoke eval on 2 slides
log "G3: smoke eval (2 slides)"
PYTHONUNBUFFERED=1 $PY -u eval_gars_cascade.py \
  stage1=$V8_CKPT \
  neighborhood_mode=true stage2_neighborhood=$V36_DIR/best_checkpoint.pt \
  thresholds=[0.5] limit_slides=2 \
  min_component_size=2 closing_iters=1 \
  label=v3.36_g3_smoke wandb.mode=disabled

# G4 — full benchmark
log "G4: full benchmark (124 slides, threshold sweep)"
PYTHONUNBUFFERED=1 $PY -u eval_gars_cascade.py \
  stage1=$V8_CKPT \
  neighborhood_mode=true stage2_neighborhood=$V36_DIR/best_checkpoint.pt \
  thresholds=[0.05,0.1,0.2,0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1 \
  label=v3.36_full_eval

log "v3.36 chain done."
