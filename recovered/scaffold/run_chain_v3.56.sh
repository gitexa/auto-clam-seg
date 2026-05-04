#!/usr/bin/env bash
# Chain v3.56 — unfrozen R50 + ViT.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold
source ./chain_lib.sh

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[v3.56 $(date +%H:%M:%S)] $*"; }

log "training v3.56 GNCAF (unfrozen R50, lr=5e-5)"
chain_run v3_56_train /tmp/v3.56.log \
  $PY -u train_gncaf_transunet.py --config-name=config_transunet_v3_56
chain_wait v3_56_train || { log "training failed"; exit 1; }
log "training done"

V56_DIR=$(ls -1dt $EXP/gars_gncaf_v3.56_gncaf_transunet_unfrozen_* 2>/dev/null | head -1)
V56_CKPT=$V56_DIR/best_checkpoint.pt
[ -s "$V56_CKPT" ] || { log "ERR: no checkpoint"; exit 1; }
log "ckpt = $V56_CKPT"

log "launching 4-shard val eval"
for SHARD in 0 1 2 3; do
  chain_run v3_56_eval_$SHARD /tmp/v3.56_eval_shard${SHARD}.log \
    $PY -u eval_gars_gncaf_transunet.py \
      checkpoint=$V56_CKPT batch_size=16 num_workers=8 \
      slide_offset=$SHARD slide_stride=4 \
      wandb.mode=disabled \
      label=v3.56_full_eval_shard${SHARD}
done
for SHARD in 0 1 2 3; do
  chain_wait v3_56_eval_$SHARD || log "shard $SHARD failed"
done

chain_run v3_56_combine /tmp/v3.56_combine.log \
  $PY combine_gncaf_shards.py --label_prefix v3.56_full_eval_shard
chain_wait v3_56_combine || log "combine failed"

log "v3.56 chain complete."
