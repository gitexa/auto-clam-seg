#!/usr/bin/env bash
# v3.37 chain: wait for v3.36 chain to finish, then verify model + smoke
# train + full train + G3/G4 cascade evals via region_mode.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
V8_CKPT=$EXP/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

# 0. Wait for v3.36 chain to finish (training + G3 + G4).
log "Waiting for v3.36 chain to finish (run_chain_v3.36.sh + eval_gars_cascade)"
until ! pgrep -f "run_chain_v3.36" >/dev/null 2>&1 && \
      ! pgrep -f "label=v3.36_full_eval" >/dev/null 2>&1 && \
      ! pgrep -f "label=v3.36_full" >/dev/null 2>&1; do
  sleep 30
done
log "v3.36 chain done — GPU should be free"

# 1. G1: model strict-load + dummy forward (no GPU needed beyond load).
log "v3.37 G1: RegionDecoder shape check"
$PY -u -c "
import torch
from region_decoder_model import RegionDecoder
m = RegionDecoder().eval()
B,P=2,9
y = m(torch.randn(B,P,3,256,256), torch.randn(B,P,1536),
      torch.randn(B,P,256), torch.ones(B,P,dtype=torch.bool))
assert y.shape == (B,3,768,768), y.shape
print('RegionDecoder forward OK', y.shape)
"

# 2. G2: 1-epoch smoke (small dataset, no wandb).
log "v3.37 G2: 1-epoch smoke"
PYTHONUNBUFFERED=1 $PY -u train_gars_region.py \
  train.epochs=1 train.batch_size=4 \
  data.max_windows_per_slide=4 \
  wandb.mode=disabled label=v3.37_smoke

# 3. Full training (30 ep).
log "v3.37 full training (30 epochs)"
PYTHONUNBUFFERED=1 nohup $PY -u train_gars_region.py \
  train.epochs=30 train.batch_size=8 \
  data.max_windows_per_slide=64 \
  label=v3.37_full > /tmp/v3.37.log 2>&1 &
sleep 60
V37_DIR=$(ls -1dt $EXP/gars_region_v3.37_full_* 2>/dev/null | head -1)
log "v3.37 dir = $V37_DIR"
until ! pgrep -f "label=v3.37_full" >/dev/null 2>&1; do sleep 30; done
log "v3.37 training done"

if [ ! -s "$V37_DIR/best_checkpoint.pt" ]; then
  log "ERR: no checkpoint in $V37_DIR"; exit 1
fi
log "ckpt = $V37_DIR/best_checkpoint.pt"

# 4. G3: 2-slide region-eval smoke
log "v3.37 G3: 2-slide eval smoke"
PYTHONUNBUFFERED=1 $PY -u eval_gars_cascade.py \
  stage1=$V8_CKPT \
  region_mode=true stage2_region=$V37_DIR/best_checkpoint.pt \
  thresholds=[0.5] limit_slides=2 \
  min_component_size=2 closing_iters=1 \
  label=v3.37_g3_smoke wandb.mode=disabled

# 5. G4: full benchmark
log "v3.37 G4: full benchmark"
PYTHONUNBUFFERED=1 $PY -u eval_gars_cascade.py \
  stage1=$V8_CKPT \
  region_mode=true stage2_region=$V37_DIR/best_checkpoint.pt \
  thresholds=[0.05,0.1,0.2,0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1 \
  label=v3.37_full_eval

log "v3.37 chain done."
