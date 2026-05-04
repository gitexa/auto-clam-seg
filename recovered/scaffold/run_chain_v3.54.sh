#!/usr/bin/env bash
# Chain v3.54 — full paper-style GNCAF training.
# Uses the marker-based chain_lib.sh (no pgrep races).
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold
source ./chain_lib.sh

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[v3.54 $(date +%H:%M:%S)] $*"; }

# ── Step 1: training ────────────────────────────────────
log "training v3.54 GNCAF full (R50+ViT pretrained, no aug, Dice, bg-only)"
chain_run v3_54_train /tmp/v3.54.log \
  $PY -u train_gncaf_transunet.py --config-name=config_transunet_v3_54
chain_wait v3_54_train || { log "training failed"; exit 1; }
log "training done"

V54_DIR=$(ls -1dt $EXP/gars_gncaf_v3.54_gncaf_transunet_full_* 2>/dev/null | head -1)
if [ ! -s "$V54_DIR/best_checkpoint.pt" ]; then
  log "ERR: no best_checkpoint.pt"; exit 1
fi
V54_CKPT=$V54_DIR/best_checkpoint.pt
log "ckpt = $V54_CKPT"

# ── Step 2: 4-shard slide-level eval ────────────────────
log "launching 4-shard val eval"
for SHARD in 0 1 2 3; do
  chain_run v3_54_eval_$SHARD /tmp/v3.54_eval_shard${SHARD}.log \
    $PY -u eval_gars_gncaf_transunet.py \
      checkpoint=$V54_CKPT batch_size=16 num_workers=8 \
      slide_offset=$SHARD slide_stride=4 \
      wandb.mode=disabled \
      label=v3.54_full_eval_shard${SHARD}
done
for SHARD in 0 1 2 3; do
  chain_wait v3_54_eval_$SHARD || log "shard $SHARD failed (continuing)"
done
log "all 4 eval shards complete"

# ── Step 3: combine ────────────────────────────────────
log "combining shards"
chain_run v3_54_combine /tmp/v3.54_combine.log \
  $PY combine_gncaf_shards.py --label_prefix v3.54_full_eval_shard
chain_wait v3_54_combine || log "combine failed"

log "v3.54 chain complete."
