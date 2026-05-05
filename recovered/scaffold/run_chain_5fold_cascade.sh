#!/usr/bin/env bash
# Chain: train cascade (Stage 1 GAT + Stage 2 RegionDecoder) on folds 1-4.
# Fold 0 already exists at:
#   Stage 1: gars_stage1_v3.8_gatv2_5hop_20260501_043445
#   Stage 2: gars_region_v3.37_full_20260502_144124
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold
source ./chain_lib.sh

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[cascade 5fold $(date +%H:%M:%S)] $*"; }

for FOLD in 1 2 3 4; do
  log "===== FOLD $FOLD ====="

  # ── Stage 1: GraphTLSDetector ────────────────────────
  log "fold $FOLD: training Stage 1 (GATv2 5-hop)"
  chain_run cascade_fold${FOLD}_s1 /tmp/cascade_fold${FOLD}_s1.log \
    $PY -u train_gars_stage1.py \
      model=gatv2_5hop \
      train.k_folds=5 train.fold_idx=$FOLD \
      label=cascade_5fold_fold${FOLD}_s1
  chain_wait cascade_fold${FOLD}_s1 || { log "fold $FOLD Stage 1 failed"; continue; }

  S1_DIR=$(ls -1dt $EXP/gars_stage1_cascade_5fold_fold${FOLD}_s1_* 2>/dev/null | head -1)
  S1_CKPT=$S1_DIR/best_checkpoint.pt
  if [ ! -s "$S1_CKPT" ]; then
    log "fold $FOLD: no Stage 1 ckpt, skipping"
    continue
  fi
  log "fold $FOLD Stage 1 ckpt = $S1_CKPT"

  # ── Stage 2: RegionDecoder ────────────────────────────
  log "fold $FOLD: training Stage 2 (RegionDecoder, frozen Stage 1)"
  chain_run cascade_fold${FOLD}_s2 /tmp/cascade_fold${FOLD}_s2.log \
    $PY -u train_gars_region.py \
      stage1=$S1_CKPT \
      train.k_folds=5 train.fold_idx=$FOLD \
      label=cascade_5fold_fold${FOLD}_s2
  chain_wait cascade_fold${FOLD}_s2 || { log "fold $FOLD Stage 2 failed"; continue; }

  S2_DIR=$(ls -1dt $EXP/gars_region_cascade_5fold_fold${FOLD}_s2_* 2>/dev/null | head -1)
  S2_CKPT=$S2_DIR/best_checkpoint.pt
  if [ ! -s "$S2_CKPT" ]; then
    log "fold $FOLD: no Stage 2 ckpt, skipping"
    continue
  fi

  # ── Cascade eval (4-shard) ────────────────────────────
  log "fold $FOLD: launching 4-shard cascade eval"
  for SHARD in 0 1 2 3; do
    chain_run cascade_fold${FOLD}_eval_${SHARD} /tmp/cascade_fold${FOLD}_eval_shard${SHARD}.log \
      $PY -u eval_gars_cascade.py \
        stage1=$S1_CKPT \
        region_mode=true stage2_region=$S2_CKPT \
        slide_offset=$SHARD slide_stride=4 \
        thresholds=[0.5] \
        fold_idx=$FOLD k_folds=5 \
        wandb.mode=disabled \
        label=cascade_5fold_fold${FOLD}_eval_shard${SHARD}
  done
  for SHARD in 0 1 2 3; do
    chain_wait cascade_fold${FOLD}_eval_${SHARD} || log "fold $FOLD shard $SHARD failed"
  done

  log "fold $FOLD complete"
done

log "===== all cascade 5fold chains done ====="
