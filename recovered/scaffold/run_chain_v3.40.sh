#!/usr/bin/env bash
# v3.40 GNCAF paper-repro chain.
#   1. Wait for v3.37 G4 124-slide eval to finish (frees GPU).
#   2. Train v3.40 GNCAF (paper-faithful 12L ViT, 60 epochs).
#   3. Slide-level 4-shard eval via eval_gars_gncaf.py.
#   4. Combine shards into one JSON.
#   5. Append result to summary.md / experiment_log.md (manual after).
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

# 1. Wait for v3.37 G4 to free GPU.
log "Waiting for v3.37 G4 124-slide eval to exit"
until ! pgrep -f "label=v3.37_full_eval_v3_124" >/dev/null 2>&1; do
  sleep 30
done
log "v3.37 G4 done — GPU should be free"

# 2. Train v3.40.
log "Launching v3.40 GNCAF training (paper-repro, 60 ep)"
PYTHONUNBUFFERED=1 nohup $PY -u train_gncaf.py \
  --config-name=config_paper_repro \
  > /tmp/v3.40.log 2>&1 &
sleep 60
V40_DIR=$(ls -1dt $EXP/gars_gncaf_v3.40_gncaf_paper_repro_* 2>/dev/null | head -1)
log "v3.40 dir = $V40_DIR"
until ! pgrep -f "label=v3.40_gncaf_paper_repro" >/dev/null 2>&1; do
  sleep 30
done
log "v3.40 training done"

if [ ! -s "$V40_DIR/best_checkpoint.pt" ]; then
  log "ERR: no v3.40 checkpoint"; exit 1
fi
log "ckpt = $V40_DIR/best_checkpoint.pt"

# 3. 4-shard slide-level eval.
log "Launching 4 parallel eval shards"
V40_CKPT=$V40_DIR/best_checkpoint.pt
for SHARD in 0 1 2 3; do
  PYTHONUNBUFFERED=1 nohup $PY -u eval_gars_gncaf.py \
    checkpoint=$V40_CKPT batch_size=32 num_workers=8 \
    slide_offset=$SHARD slide_stride=4 \
    label=v3.40_full_eval_shard${SHARD} \
    wandb.mode=disabled \
    > /tmp/v3.40_eval_shard${SHARD}.log 2>&1 &
done
sleep 120
log "Waiting for all 4 eval shards to finish"
until [ "$(pgrep -f 'eval_gars_gncaf.*v3.40_full_eval_shard' | wc -l)" -eq 0 ]; do
  sleep 30
done
log "All 4 shards done"

# 4. Combine.
log "Combining shard results"
$PY combine_gncaf_shards.py --label_prefix v3.40_full_eval_shard

log "v3.40 chain done. Inspect /home/ubuntu/ahaas-persistent-std-tcga/experiments/v3.40_combined.json"
