#!/usr/bin/env bash
# Chain: train GCUNet (v3.59) on folds 1, 2, 3, 4 sequentially.
# Fold 0 trains separately first (gars_gncaf_v3.59_fold0_*).
# v3.59 = GCUNet-faithful GCN aggregation (concat-all-hops, paper Eq.3).
# DECISION GATE — only launch this chain AFTER v3.59 fold-0 mDice
# is comparable to or above v3.58 fold-0 (no architectural regression).
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold
source ./chain_lib.sh

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[v3.59 5fold $(date +%H:%M:%S)] $*"; }

for FOLD in 1 2 3 4; do
  log "===== FOLD $FOLD ====="

  log "training v3.59 fold $FOLD"
  chain_run v3_59_fold${FOLD}_train /tmp/v3.59_fold${FOLD}.log \
    $PY -u train_gncaf_transunet.py \
      --config-name=config_transunet_v3_59 \
      fold_idx=$FOLD k_folds=5 \
      label=v3.59_5fold_fold${FOLD}
  chain_wait v3_59_fold${FOLD}_train || { log "fold $FOLD train failed"; continue; }

  V58_DIR=$(ls -1dt $EXP/gars_gncaf_v3.59_5fold_fold${FOLD}_* 2>/dev/null | head -1)
  V58_CKPT=$V58_DIR/best_checkpoint.pt
  if [ ! -s "$V58_CKPT" ]; then
    log "fold $FOLD: no checkpoint, skipping eval"
    continue
  fi
  log "fold $FOLD ckpt = $V58_CKPT"

  log "fold $FOLD: launching 4-shard eval"
  for SHARD in 0 1 2 3; do
    chain_run v3_59_fold${FOLD}_eval_${SHARD} /tmp/v3.59_fold${FOLD}_eval_shard${SHARD}.log \
      $PY -u eval_gars_gncaf_transunet.py \
        checkpoint=$V58_CKPT batch_size=8 num_workers=8 \
        slide_offset=$SHARD slide_stride=4 \
        fold_idx=$FOLD k_folds=5 \
        wandb.mode=disabled \
        label=v3.59_5fold_fold${FOLD}_eval_shard${SHARD}
  done
  for SHARD in 0 1 2 3; do
    chain_wait v3_59_fold${FOLD}_eval_${SHARD} || log "fold $FOLD shard $SHARD failed"
  done

  log "fold $FOLD: combining shards"
  chain_run v3_59_fold${FOLD}_combine /tmp/v3.59_fold${FOLD}_combine.log \
    $PY combine_gncaf_shards.py --label_prefix v3.59_5fold_fold${FOLD}_eval_shard
  chain_wait v3_59_fold${FOLD}_combine || log "combine failed"

  log "fold $FOLD complete"
done

log "===== all v3.59 5fold chains done ====="
