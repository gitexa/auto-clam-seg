#!/usr/bin/env bash
# Chain: wait for v3.51 training, then 4-shard slide-level eval, then combine.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[chain v3.51 $(date +%H:%M:%S)] $*"; }

log "Waiting for v3.51 training to finish"
until ! pgrep -f "config_transunet_v3_51" >/dev/null 2>&1; do
  sleep 60
done
log "v3.51 training done"

V51_DIR=$(ls -1dt $EXP/gars_gncaf_v3.51_gncaf_transunet_bgneg_* 2>/dev/null | head -1)
log "v3.51 dir = $V51_DIR"
if [ ! -s "$V51_DIR/best_checkpoint.pt" ]; then
  log "ERR: no v3.51 checkpoint"; exit 1
fi
V51_CKPT=$V51_DIR/best_checkpoint.pt
log "ckpt = $V51_CKPT"

log "Launching v3.51 4-shard eval"
for SHARD in 0 1 2 3; do
  PYTHONUNBUFFERED=1 nohup $PY -u eval_gars_gncaf_transunet.py \
    checkpoint=$V51_CKPT batch_size=16 num_workers=8 \
    slide_offset=$SHARD slide_stride=4 \
    wandb.mode=disabled \
    label=v3.51_full_eval_shard${SHARD} \
    > /tmp/v3.51_eval_shard${SHARD}.log 2>&1 &
done
sleep 60
until [ "$(pgrep -f 'eval_gars_gncaf_transunet.*v3.51_full_eval' | wc -l)" -eq 0 ]; do
  sleep 30
done
log "v3.51 eval shards done"

log "Combining v3.51 shards"
$PY combine_gncaf_shards.py --label_prefix v3.51_full_eval_shard || true

log "v3.51 chain done."
