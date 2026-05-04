#!/usr/bin/env bash
# Chain: wait for v4.0 cascade-eval shards, then fire v4.1 (hard top-K) +
# its cascade eval. Hard mode = no soft-weighting; baseline test of "is
# the soft-weighting the differentiator, or just joint-training itself?"
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[chain v4.1 $(date +%H:%M:%S)] $*"; }

log "Waiting for v4.0 val eval to finish"
while pgrep -f "v4.0_val_eval_shard" >/dev/null 2>&1; do
  sleep 30
done
log "v4.0 val eval done"

log "Launching v4.1 e2e training (hard top-K, no soft weighting)"
PYTHONUNBUFFERED=1 nohup $PY -u train_e2e_v4.py \
  --config-name=config_v4_0 \
  label=v4.1_e2e_hard \
  train.e2e_mode=hard \
  > /tmp/v4.1.log 2>&1 &
sleep 60
V41_DIR=$(ls -1dt $EXP/gars_e2e_v4.1_e2e_hard_* 2>/dev/null | head -1)
log "v4.1 dir = $V41_DIR"

until ! pgrep -f "label=v4.1_e2e_hard" >/dev/null 2>&1; do
  sleep 60
done
log "v4.1 training done"

V41_CKPT=$V41_DIR/best_checkpoint.pt
if [ ! -s "$V41_CKPT" ]; then
  log "ERR: no v4.1 checkpoint"; exit 1
fi

log "Extracting Stage 1 + Stage 2 from v4.1 joint ckpt"
$PY -c "
import torch
ckpt = torch.load('$V41_CKPT', map_location='cpu', weights_only=False)
s1 = {'model_state_dict': ckpt['stage1_state_dict'],
      'config': {'model': {'in_dim': 1536, 'hidden_dim': 256, 'n_hops': 5,
                           'gnn_type': 'gatv2', 'dropout': 0.1, 'gat_heads': 4}},
      'epoch': ckpt['epoch'], 'best_mDice': ckpt['best_mDice']}
s2 = {'model_state_dict': ckpt['stage2_state_dict'],
      'config': {'model': {'uni_dim': 1536, 'gat_dim': 256, 'hidden_channels': 64,
                           'n_classes': 3, 'grid_n': 3, 'rgb_pretrained': True,
                           'freeze_rgb_encoder': False}},
      'epoch': ckpt['epoch'], 'best_mDice': ckpt['best_mDice']}
torch.save(s1, '$V41_DIR/stage1_v4.1.pt')
torch.save(s2, '$V41_DIR/stage2_v4.1.pt')
print('Extracted Stage 1 + Stage 2')
"

log "Launching v4.1 cascade val eval (4 shards)"
for SHARD in 0 1 2 3; do
  PYTHONUNBUFFERED=1 nohup $PY -u eval_gars_cascade.py \
    stage1=$V41_DIR/stage1_v4.1.pt \
    region_mode=true stage2_region=$V41_DIR/stage2_v4.1.pt \
    slide_offset=$SHARD slide_stride=4 \
    thresholds=[0.5] \
    wandb.mode=disabled \
    label=v4.1_val_eval_shard${SHARD} \
    > /tmp/v4.1_val_shard${SHARD}.log 2>&1 &
done
sleep 60
until [ "$(pgrep -f 'v4.1_val_eval_shard' | wc -l)" -eq 0 ]; do
  sleep 30
done
log "v4.1 val eval done"

log "all v4.x chains done."
