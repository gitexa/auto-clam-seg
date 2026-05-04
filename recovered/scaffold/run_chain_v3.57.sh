#!/usr/bin/env bash
# Chain v3.57 — 12-layer ViT (paper-faithful depth).
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold
source ./chain_lib.sh

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[v3.57 $(date +%H:%M:%S)] $*"; }

log "training v3.57 GNCAF (12-layer ViT, unfrozen R50)"
chain_run v3_57_train /tmp/v3.57.log \
  $PY -u train_gncaf_transunet.py --config-name=config_transunet_v3_57
chain_wait v3_57_train || { log "training failed"; exit 1; }

V57_DIR=$(ls -1dt $EXP/gars_gncaf_v3.57_gncaf_transunet_vit12_* 2>/dev/null | head -1)
V57_CKPT=$V57_DIR/best_checkpoint.pt
[ -s "$V57_CKPT" ] || { log "ERR: no checkpoint"; exit 1; }
log "ckpt = $V57_CKPT"

log "launching 4-shard val eval"
for SHARD in 0 1 2 3; do
  chain_run v3_57_eval_$SHARD /tmp/v3.57_eval_shard${SHARD}.log \
    $PY -u eval_gars_gncaf_transunet.py \
      checkpoint=$V57_CKPT batch_size=8 num_workers=8 \
      slide_offset=$SHARD slide_stride=4 \
      wandb.mode=disabled \
      label=v3.57_full_eval_shard${SHARD}
done
for SHARD in 0 1 2 3; do
  chain_wait v3_57_eval_$SHARD || log "shard $SHARD failed"
done

chain_run v3_57_combine /tmp/v3.57_combine.log \
  $PY combine_gncaf_shards.py --label_prefix v3.57_full_eval_shard
chain_wait v3_57_combine || log "combine failed"

log "v3.57 chain complete."
