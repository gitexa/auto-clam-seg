#!/usr/bin/env bash
# Chain: wait for v3.53 to finish, then fire v4.0 e2e training.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[chain v4.0 $(date +%H:%M:%S)] $*"; }

log "Waiting for v3.53 chain to finish"
while pgrep -f "config_transunet_v3_53" >/dev/null 2>&1 \
   || pgrep -f "v3.53_full_eval" >/dev/null 2>&1; do
  sleep 30
done
log "v3.53 chain done"

log "Launching v4.0 e2e training (soft weighting)"
PYTHONUNBUFFERED=1 nohup $PY -u train_e2e_v4.py \
  --config-name=config_v4_0 \
  > /tmp/v4.0.log 2>&1 &
sleep 60
V40_DIR=$(ls -1dt $EXP/gars_e2e_v4.0_e2e_soft_* 2>/dev/null | head -1)
log "v4.0 dir = $V40_DIR"

until ! pgrep -f "config_v4_0" >/dev/null 2>&1; do
  sleep 60
done
log "v4.0 training done"

# Optional follow-ups: hard variant, larger K, etc.
if [ -d "$V40_DIR" ] && [ -s "$V40_DIR/best_checkpoint.pt" ]; then
  log "Launching v4.1 (hard top-K, sanity baseline)"
  PYTHONUNBUFFERED=1 nohup $PY -u train_e2e_v4.py \
    --config-name=config_v4_0 \
    label=v4.1_e2e_hard train.e2e_mode=hard \
    > /tmp/v4.1.log 2>&1 &
  sleep 60
  until ! pgrep -f "label=v4.1_e2e_hard" >/dev/null 2>&1; do
    sleep 60
  done
  log "v4.1 done"
fi

log "all v4 chains done."
