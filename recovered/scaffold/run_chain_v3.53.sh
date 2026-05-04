#!/usr/bin/env bash
# Chain: train v3.53 (ImageNet R50 + augment + dice), then 4-shard val eval.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[chain v3.53 $(date +%H:%M:%S)] $*"; }

log "Launching v3.53 GNCAF + ImageNet R50 + augment + Dice"
PYTHONUNBUFFERED=1 nohup $PY -u train_gncaf_transunet.py \
  --config-name=config_transunet_v3_53 \
  > /tmp/v3.53.log 2>&1 &
sleep 60

V53_DIR=$(ls -1dt $EXP/gars_gncaf_v3.53_gncaf_transunet_imagenetR50_aug_dice_* 2>/dev/null | head -1)
log "v3.53 dir = $V53_DIR"

until ! pgrep -f "config_transunet_v3_53" >/dev/null 2>&1; do
  sleep 60
done
log "v3.53 training done"

V53_CKPT=$V53_DIR/best_checkpoint.pt
if [ ! -s "$V53_CKPT" ]; then
  log "ERR: no v3.53 checkpoint"; exit 1
fi
log "ckpt = $V53_CKPT"

log "Launching v3.53 4-shard val eval"
for SHARD in 0 1 2 3; do
  PYTHONUNBUFFERED=1 nohup $PY -u eval_gars_gncaf_transunet.py \
    checkpoint=$V53_CKPT batch_size=16 num_workers=8 \
    slide_offset=$SHARD slide_stride=4 \
    wandb.mode=disabled \
    label=v3.53_full_eval_shard${SHARD} \
    > /tmp/v3.53_eval_shard${SHARD}.log 2>&1 &
done
sleep 60
until [ "$(pgrep -f 'eval_gars_gncaf_transunet.*v3.53_full_eval' | wc -l)" -eq 0 ]; do
  sleep 30
done
log "v3.53 eval shards done"

log "Combining v3.53 shards"
$PY combine_gncaf_shards.py --label_prefix v3.53_full_eval_shard || true

log "v3.53 chain done."
