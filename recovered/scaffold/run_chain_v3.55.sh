#!/usr/bin/env bash
# Chain v3.55 — stable low-LR GNCAF training. Uses chain_lib.sh.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold
source ./chain_lib.sh

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[v3.55 $(date +%H:%M:%S)] $*"; }

log "training v3.55 GNCAF (lr=5e-5, grad_clip=0.5)"
chain_run v3_55_train /tmp/v3.55.log \
  $PY -u train_gncaf_transunet.py --config-name=config_transunet_v3_55
chain_wait v3_55_train || { log "training failed"; exit 1; }
log "training done"

V55_DIR=$(ls -1dt $EXP/gars_gncaf_v3.55_gncaf_transunet_lowlr_* 2>/dev/null | head -1)
V55_CKPT=$V55_DIR/best_checkpoint.pt
[ -s "$V55_CKPT" ] || { log "ERR: no checkpoint"; exit 1; }
log "ckpt = $V55_CKPT"

log "launching 4-shard val eval"
for SHARD in 0 1 2 3; do
  chain_run v3_55_eval_$SHARD /tmp/v3.55_eval_shard${SHARD}.log \
    $PY -u eval_gars_gncaf_transunet.py \
      checkpoint=$V55_CKPT batch_size=16 num_workers=8 \
      slide_offset=$SHARD slide_stride=4 \
      wandb.mode=disabled \
      label=v3.55_full_eval_shard${SHARD}
done
for SHARD in 0 1 2 3; do
  chain_wait v3_55_eval_$SHARD || log "shard $SHARD failed (continuing)"
done
log "all 4 eval shards complete"

log "combining shards"
chain_run v3_55_combine /tmp/v3.55_combine.log \
  $PY combine_gncaf_shards.py --label_prefix v3.55_full_eval_shard
chain_wait v3_55_combine || log "combine failed"

log "v3.55 chain complete."
