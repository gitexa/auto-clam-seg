#!/bin/bash
# Sequential 5-fold chain for v3.62 paper-strict GNCAF.
# Fold 0 already trained at gars_gncaf_v3.62_fold0_20260509_143857.
# This script trains folds 1, 2, 3, 4 sequentially, then runs eval per fold.
set -e
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP_ROOT=/home/ubuntu/ahaas-persistent-std-tcga/experiments

for fold in 1 2 3 4; do
  echo "============================================================"
  echo "===  v3.62 fold $fold — TRAIN  ==="
  echo "============================================================"
  WANDB_MODE=online $PY train_gncaf_transunet.py \
    --config-name=config_transunet_v3_62 \
    fold_idx=$fold \
    label=v3.62_fold${fold} \
    > /tmp/v3_62_fold${fold}_train.log 2>&1

  TRAIN_DIR=$(ls -td ${EXP_ROOT}/gars_gncaf_v3.62_fold${fold}_* 2>/dev/null | head -1)
  if [ -z "$TRAIN_DIR" ]; then
    echo "ERROR: train dir for fold $fold not found"
    exit 1
  fi
  echo "Trained: $TRAIN_DIR"

  echo "===  v3.62 fold $fold — EVAL (4-shard) ==="
  for shard in 0 1 2 3; do
    WANDB_MODE=disabled $PY eval_gars_gncaf_transunet.py \
      checkpoint=${TRAIN_DIR}/best_checkpoint.pt \
      label=v3.62_union_fullcohort_fold${fold}_shard${shard} \
      fold_idx=$fold \
      slide_offset=$shard \
      slide_stride=4 \
      num_workers=4 \
      > /tmp/v3_62_fold${fold}_eval_shard${shard}.log 2>&1 &
  done
  wait
  echo "Fold $fold eval done."
done

echo "============================================================"
echo "===  v3.62 5-fold complete  ==="
echo "============================================================"
