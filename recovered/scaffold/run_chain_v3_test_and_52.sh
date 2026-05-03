#!/usr/bin/env bash
# Chain: launch v3.52 RegionDecoder training + v3.37 cascade test eval (4 shards)
# in parallel. Combine test-eval shards. Then auto-fire v3.52 4-shard eval after
# v3.52 training finishes.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
STAGE1=/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt
V37_CKPT=/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_region_v3.37_full_20260502_144124/best_checkpoint.pt
log() { echo "[chain $(date +%H:%M:%S)] $*"; }

# ── v3.37 test-set 4-shard eval ───────────────────────────
log "Launching v3.37 cascade test-set eval (4 shards)"
for SHARD in 0 1 2 3; do
  PYTHONUNBUFFERED=1 nohup $PY -u eval_gars_cascade.py \
    stage1=$STAGE1 \
    region_mode=true stage2_region=$V37_CKPT \
    use_test_split=true \
    slide_offset=$SHARD slide_stride=4 \
    thresholds=[0.5] \
    wandb.mode=disabled \
    label=v3.37_test_eval_shard${SHARD} \
    > /tmp/v3.37_test_shard${SHARD}.log 2>&1 &
done

# ── v3.52 RegionDecoder training ──────────────────────────
sleep 5
log "Launching v3.52 RegionDecoder + bg-only training"
PYTHONUNBUFFERED=1 nohup $PY -u train_gars_region.py \
  --config-name=config_v3_52 \
  > /tmp/v3.52.log 2>&1 &

# ── Wait for test-eval shards ─────────────────────────────
log "Waiting for v3.37 test-eval shards"
sleep 60
until [ "$(pgrep -f 'eval_gars_cascade.*v3.37_test_eval' | wc -l)" -eq 0 ]; do
  sleep 30
done
log "v3.37 test-eval shards done"

# Combine test-eval — cascade has different combine logic than gncaf;
# write inline collator.
$PY -c "
import json, glob
from pathlib import Path
EXP = Path('$EXP')
dirs = sorted(EXP.glob('gars_cascade_v3.37_test_eval_shard*_*'))
print('Found', len(dirs), 'shard dirs')
import pandas as pd
import numpy as np
rows = []
for d in dirs:
    csvs = list(d.glob('*.csv'))
    for c in csvs:
        try:
            df = pd.read_csv(c)
            if 'thr' in df.columns:
                df = df[df['thr']==0.5]
            rows.append(df)
        except Exception as e:
            print('skip', c, e)
if rows:
    full = pd.concat(rows, ignore_index=True)
    full = full.drop_duplicates(subset=['slide_id'], keep='first') if 'slide_id' in full.columns else full
    full.to_csv(EXP / 'v3.37_test_combined.csv', index=False)
    # aggregate
    print('combined slides:', len(full))
    if 'mDice_pix' in full.columns:
        print(f'mean mDice_pix:  {full[\"mDice_pix\"].mean():.4f}')
    for col in ['tls_dice_pix','gc_dice_pix','tls_dice_grid','gc_dice_grid']:
        if col in full.columns:
            print(f'mean {col}: {full[col].mean():.4f}')
" 2>&1 | tee -a /tmp/chain_test_v52.log

# ── Wait for v3.52 training ───────────────────────────────
log "Waiting for v3.52 training"
until ! pgrep -f "config_v3_52" >/dev/null 2>&1; do
  sleep 60
done
log "v3.52 training done"

V52_DIR=$(ls -1dt $EXP/gars_region_v3.52_region_bgneg_* 2>/dev/null | head -1)
log "v3.52 dir = $V52_DIR"
V52_CKPT=$V52_DIR/best_checkpoint.pt

# ── v3.52 cascade eval, fold-0 val + test ─────────────────
if [ -s "$V52_CKPT" ]; then
  log "Launching v3.52 cascade eval — fold-0 val (4 shards)"
  for SHARD in 0 1 2 3; do
    PYTHONUNBUFFERED=1 nohup $PY -u eval_gars_cascade.py \
      stage1=$STAGE1 \
      region_mode=true stage2_region=$V52_CKPT \
      slide_offset=$SHARD slide_stride=4 \
      thresholds=[0.5] \
      wandb.mode=disabled \
      label=v3.52_val_eval_shard${SHARD} \
      > /tmp/v3.52_val_shard${SHARD}.log 2>&1 &
  done
  sleep 60
  until [ "$(pgrep -f 'eval_gars_cascade.*v3.52_val_eval' | wc -l)" -eq 0 ]; do
    sleep 30
  done
  log "v3.52 val eval shards done"

  log "Launching v3.52 cascade eval — TEST (4 shards)"
  for SHARD in 0 1 2 3; do
    PYTHONUNBUFFERED=1 nohup $PY -u eval_gars_cascade.py \
      stage1=$STAGE1 \
      region_mode=true stage2_region=$V52_CKPT \
      use_test_split=true \
      slide_offset=$SHARD slide_stride=4 \
      thresholds=[0.5] \
      wandb.mode=disabled \
      label=v3.52_test_eval_shard${SHARD} \
      > /tmp/v3.52_test_shard${SHARD}.log 2>&1 &
  done
  sleep 60
  until [ "$(pgrep -f 'eval_gars_cascade.*v3.52_test_eval' | wc -l)" -eq 0 ]; do
    sleep 30
  done
  log "v3.52 test eval shards done"
else
  log "WARN: no v3.52 checkpoint, skipping eval"
fi

log "all chains done."
