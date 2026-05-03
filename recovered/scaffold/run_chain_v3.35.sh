#!/usr/bin/env bash
# Wait for v3.35 GNCAF training, then run slide-level eval.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

V35_DIR=$(ls -1dt $EXP/gars_gncaf_v3.35_gncaf_full_12layer_* 2>/dev/null | head -1)
log "v3.35 dir = $V35_DIR"

log "Waiting for v3.35 training to finish"
until ! pgrep -f "python.*--config-name=config_full" >/dev/null 2>&1; do sleep 30; done
log "v3.35 training done"

if [ ! -s "$V35_DIR/best_checkpoint.pt" ]; then
  log "ERR: no v3.35 checkpoint"; exit 1
fi
log "checkpoint = $V35_DIR/best_checkpoint.pt"

log "Launching slide-level eval (~91 min)"
PYTHONUNBUFFERED=1 $PY -u eval_gars_gncaf.py \
  checkpoint=$V35_DIR/best_checkpoint.pt \
  num_workers=8 batch_size=16 \
  label=v3.35_full_eval

log "v3.35 chain done."
