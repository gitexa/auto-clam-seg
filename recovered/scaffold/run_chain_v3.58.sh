#!/usr/bin/env bash
# Chain v3.58 — 12-layer ViT + augmentation.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold
source ./chain_lib.sh

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[v3.58 $(date +%H:%M:%S)] $*"; }

log "training v3.58 GNCAF (12-layer ViT + augmentation, lr=5e-5)"
chain_run v3_58_train /tmp/v3.58.log \
  $PY -u train_gncaf_transunet.py --config-name=config_transunet_v3_58
chain_wait v3_58_train || { log "training failed"; exit 1; }

V58_DIR=$(ls -1dt $EXP/gars_gncaf_v3.58_gncaf_transunet_vit12_aug_* 2>/dev/null | head -1)
V58_CKPT=$V58_DIR/best_checkpoint.pt
[ -s "$V58_CKPT" ] || { log "ERR: no checkpoint"; exit 1; }
log "ckpt = $V58_CKPT"

log "launching 4-shard val eval"
for SHARD in 0 1 2 3; do
  chain_run v3_58_eval_$SHARD /tmp/v3.58_eval_shard${SHARD}.log \
    $PY -u eval_gars_gncaf_transunet.py \
      checkpoint=$V58_CKPT batch_size=8 num_workers=8 \
      slide_offset=$SHARD slide_stride=4 \
      wandb.mode=disabled \
      label=v3.58_full_eval_shard${SHARD}
done
for SHARD in 0 1 2 3; do
  chain_wait v3_58_eval_$SHARD || log "shard $SHARD failed"
done

chain_run v3_58_combine /tmp/v3.58_combine.log \
  $PY combine_gncaf_shards.py --label_prefix v3.58_full_eval_shard
chain_wait v3_58_combine || log "combine failed"

log "v3.58 chain complete."
