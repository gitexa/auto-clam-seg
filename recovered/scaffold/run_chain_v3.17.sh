#!/usr/bin/env bash
# v3.17 sprint: n_hops=4 Stage 1 + cascade eval re-pairs.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
V0_CKPT=$EXP/gars_stage1_v3.0_gatv2_3hop_20260430_002624/best_checkpoint.pt
V8_CKPT=$EXP/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt
V4B_CKPT=$EXP/gars_stage2_v3.4b_h128_min4096_20260430_090510/best_checkpoint.pt
V11_DIR=$(ls -1dt $EXP/gars_stage2_v3.11_h128_min8192_* | head -1)
V12_DIR=$(ls -1dt $EXP/gars_stage2_v3.12_h128_gc_focus_* | head -1)
V11_CKPT=$V11_DIR/best_checkpoint.pt
V12_CKPT=$V12_DIR/best_checkpoint.pt

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

wait_for_ckpt () {
  local p=$1
  log "Waiting for $p"
  until [ -s "$p" ]; do sleep 30; done
  log "Found $p"
}

wait_for_label_dead () {
  local pat=$1
  log "Waiting for python with label '$pat' to exit"
  until ! pgrep -f "python.*label=$pat" >/dev/null 2>&1; do sleep 30; done
  log "label '$pat' done"
}

run_eval () {
  local name=$1; local s1=$2; local s2=$3; shift 3
  log "Cascade eval: $name (extra: $*)"
  PYTHONUNBUFFERED=1 $PY -u eval_gars_cascade.py \
    stage1=$s1 stage2=$s2 label=$name "$@"
}

# Step 1 — train v3.17 Stage 1 (n_hops=4)
log "Launching v3.17 Stage 1 (gatv2_4hop)"
PYTHONUNBUFFERED=1 nohup $PY -u train_gars_stage1.py \
  model=gatv2_4hop label=v3.17_gatv2_4hop \
  > /tmp/v3.17.log 2>&1 &
log "v3.17 PID=$!"
sleep 60
V17_DIR=$(ls -1dt $EXP/gars_stage1_v3.17_gatv2_4hop_* 2>/dev/null | head -1)
log "v3.17 dir = $V17_DIR"
wait_for_ckpt "$V17_DIR/best_checkpoint.pt"
wait_for_label_dead "v3.17_gatv2_4hop"

# Step 2 — cascade eval v3.17 + v3.4b (test n_hops=4 at deployment)
run_eval v3.23_s1v3.17_s2v3.4b \
  $V17_DIR/best_checkpoint.pt $V4B_CKPT \
  thresholds=[0.05,0.1,0.2,0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1

# Step 3 — re-pair v3.8 Stage 1 with min8192 Stage 2
run_eval v3.24_s1v3.8_s2v3.11 \
  $V8_CKPT $V11_CKPT \
  thresholds=[0.2,0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1

# Step 4 — re-pair v3.8 Stage 1 with GC-focused Stage 2
run_eval v3.25_s1v3.8_s2v3.12 \
  $V8_CKPT $V12_CKPT \
  thresholds=[0.2,0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1

log "v3.17 chain done."
