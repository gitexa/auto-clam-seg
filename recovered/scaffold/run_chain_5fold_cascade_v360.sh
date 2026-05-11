#!/usr/bin/env bash
# Chain: train v3.60 multi-scale bipartite cascade on folds 1-4.
# Fold 0 already exists at:
#   Stage 1 (multi-scale): gars_stage1_v3.60_5fold_fold0_s1_20260507_001638
#   Stage 2 (paired):       gars_region_v3.60_fold0_s2_paired_20260507_030713
#                           (or whichever directory matches if relaunched)
#
# DECISION GATE — only run this AFTER fold-0 paired Stage 2 retrain
# clears mDice_pix ≥ 0.86 vs the v3.37 baseline (0.845). Otherwise
# document as no-op and stop.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold
source ./chain_lib.sh

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[v3.60 5fold $(date +%H:%M:%S)] $*"; }

for FOLD in 1 2 3 4; do
  log "===== FOLD $FOLD ====="

  # ── Stage 1: MultiScaleGraphTLSDetector (bipartite GAT) ───
  log "fold $FOLD: training Stage 1 (multi-scale 5-hop GATv2)"
  chain_run v360_fold${FOLD}_s1 /tmp/v360_fold${FOLD}_s1.log \
    $PY -u train_gars_stage1_multiscale.py \
      model=gatv2_5hop_multiscale \
      train.k_folds=5 train.fold_idx=$FOLD \
      label=v3.60_5fold_fold${FOLD}_s1
  chain_wait v360_fold${FOLD}_s1 || { log "fold $FOLD Stage 1 failed"; continue; }

  S1_DIR=$(ls -1dt $EXP/gars_stage1_v3.60_5fold_fold${FOLD}_s1_* 2>/dev/null | head -1)
  S1_CKPT=$S1_DIR/best_checkpoint.pt
  if [ ! -s "$S1_CKPT" ]; then
    log "fold $FOLD: no Stage 1 ckpt, skipping"
    continue
  fi
  log "fold $FOLD Stage 1 ckpt = $S1_CKPT"

  # ── Stage 2: RegionDecoder (paired with multi-scale Stage 1) ───
  log "fold $FOLD: training Stage 2 (RegionDecoder)"
  chain_run v360_fold${FOLD}_s2 /tmp/v360_fold${FOLD}_s2.log \
    $PY -u train_gars_region.py \
      stage1=$S1_CKPT \
      train.k_folds=5 train.fold_idx=$FOLD \
      label=v3.60_5fold_fold${FOLD}_s2_paired
  chain_wait v360_fold${FOLD}_s2 || { log "fold $FOLD Stage 2 failed"; continue; }

  S2_DIR=$(ls -1dt $EXP/gars_region_v3.60_5fold_fold${FOLD}_s2_paired_* 2>/dev/null | head -1)
  S2_CKPT=$S2_DIR/best_checkpoint.pt
  if [ ! -s "$S2_CKPT" ]; then
    log "fold $FOLD: no Stage 2 ckpt, skipping"
    continue
  fi

  # ── Cascade eval (4-shard) ────────────────────────────
  log "fold $FOLD: launching 4-shard cascade eval"
  for SHARD in 0 1 2 3; do
    chain_run v360_fold${FOLD}_eval_${SHARD} /tmp/v360_fold${FOLD}_eval_shard${SHARD}.log \
      $PY -u eval_gars_cascade.py \
        stage1=$S1_CKPT \
        region_mode=true stage2_region=$S2_CKPT \
        slide_offset=$SHARD slide_stride=4 \
        thresholds=[0.5] \
        fold_idx=$FOLD k_folds=5 \
        wandb.mode=disabled \
        label=v3.60_5fold_fold${FOLD}_eval_shard${SHARD}
  done
  for SHARD in 0 1 2 3; do
    chain_wait v360_fold${FOLD}_eval_${SHARD} || log "fold $FOLD shard $SHARD failed"
  done

  log "fold $FOLD complete"
done

log "===== all v3.60 5fold chains done ====="
