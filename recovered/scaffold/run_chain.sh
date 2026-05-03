#!/usr/bin/env bash
# Autoresearch chained launcher.
# Plan:
#   1. wait for v3.8 best_checkpoint
#   2. cascade-eval v3.7e: v3.0 + v3.4b + abstain top-K (tests abstain)
#   3. cascade-eval v3.13: v3.8 + v3.4b (tests deeper graph)
#   4. wait for GPU to be free of v3.8 training
#   5. launch v3.10 Stage 1 (gatv2_3hop_h512)
#   6. wait for v3.10 best_checkpoint
#   7. cascade-eval v3.14: v3.10 + v3.4b (tests wider graph)
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

PY=/home/ubuntu/profile-clam/.venv/bin/python
EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
V38_DIR=$EXP/gars_stage1_v3.8_gatv2_5hop_20260501_043445
V0_CKPT=$EXP/gars_stage1_v3.0_gatv2_3hop_20260430_002624/best_checkpoint.pt
V4B_CKPT=$EXP/gars_stage2_v3.4b_h128_min4096_20260430_090510/best_checkpoint.pt

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

wait_for_ckpt () {
  local p=$1
  log "Waiting for $p"
  until [ -s "$p" ]; do sleep 30; done
  log "Found $p"
}

wait_for_pid_dead () {
  local pat=$1
  log "Waiting for python processes matching label '$pat' to exit"
  until ! pgrep -f "python.*label=$pat" >/dev/null 2>&1; do sleep 30; done
  log "All python processes with label '$pat' exited"
}

run_eval () {
  local name=$1; local s1=$2; local s2=$3; shift 3
  log "Launching cascade eval: $name (extra args: $*)"
  PYTHONUNBUFFERED=1 $PY -u eval_gars_cascade.py \
    stage1=$s1 stage2=$s2 label=$name \
    "$@"
}

# Step 1
wait_for_ckpt "$V38_DIR/best_checkpoint.pt"

# Step 2: cascade with abstain top-K (v3.0 + v3.4b)
# Wait until v3.8 training finishes (frees GPU) before evals.
wait_for_pid_dead "v3.8_gatv2_5hop"

run_eval v3.7e_abstain_topk \
  $V0_CKPT $V4B_CKPT \
  thresholds=[0.5] top_k_frac=0.005 top_k_abstain_thr=0.5 \
  min_component_size=2 closing_iters=1

# Step 3: cascade with v3.8 Stage 1
run_eval v3.13_s1v3.8_s2v3.4b \
  $V38_DIR/best_checkpoint.pt $V4B_CKPT \
  thresholds=[0.05,0.1,0.2,0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1

# Step 5: launch v3.10 Stage 1 (h=512)
log "Launching v3.10 Stage 1 (gatv2_3hop_h512)"
PYTHONUNBUFFERED=1 nohup $PY -u train_gars_stage1.py \
  model=gatv2_3hop_h512 label=v3.10_gatv2_3hop_h512 \
  > /tmp/v3.10.log 2>&1 &
V10_PID=$!
log "v3.10 PID=$V10_PID"

# Step 6: wait for v3.10 best_checkpoint
sleep 60
V10_DIR=$(ls -1dt $EXP/gars_stage1_v3.10_gatv2_3hop_h512_* 2>/dev/null | head -1)
if [ -z "$V10_DIR" ]; then
  log "ERR: no v3.10 dir"
  exit 1
fi
log "v3.10 dir = $V10_DIR"
wait_for_ckpt "$V10_DIR/best_checkpoint.pt"
wait_for_pid_dead "v3.10_gatv2_3hop_h512"

# Step 7: cascade with v3.10 Stage 1
run_eval v3.14_s1v3.10_s2v3.4b \
  $V10_DIR/best_checkpoint.pt $V4B_CKPT \
  thresholds=[0.05,0.1,0.2,0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1

# Step 8: train Stage 2 v3.11 with min_tls_pixels=8192 cache
# (cache must already be built — run tls_patch_dataset.py --min_tls_pixels 8192)
log "Waiting for min8192 patch cache"
until [ -s /home/ubuntu/local_data/tls_patch_dataset_min8192.pt ]; do sleep 60; done
log "Found min8192 cache"

log "Launching v3.11 Stage 2 (h=128, min_tls_pixels=8192)"
PYTHONUNBUFFERED=1 nohup $PY -u train_gars_stage2.py \
  model=univ2_decoder_h128 label=v3.11_h128_min8192 \
  cache_path=/home/ubuntu/local_data/tls_patch_dataset_min8192.pt \
  train.class_weights=[1.0,5.0,3.0] train.gc_dice_weight=1.0 \
  > /tmp/v3.11.log 2>&1 &
V11_PID=$!
log "v3.11 PID=$V11_PID"
sleep 60
V11_DIR=$(ls -1dt $EXP/gars_stage2_v3.11_h128_min8192_* 2>/dev/null | head -1)
log "v3.11 dir = $V11_DIR"
wait_for_ckpt "$V11_DIR/best_checkpoint.pt"
wait_for_pid_dead "v3.11_h128_min8192"

# Step 9: cascade with v3.11 Stage 2 + v3.0 Stage 1.
run_eval v3.15_s1v3.0_s2v3.11 \
  $V0_CKPT $V11_DIR/best_checkpoint.pt \
  thresholds=[0.05,0.1,0.2,0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1

# Step 10: train Stage 2 v3.12 with stronger GC class weight to lift GC dice.
log "Launching v3.12 Stage 2 (h=128, class_weights=[1,5,5], gc_dice=2)"
PYTHONUNBUFFERED=1 nohup $PY -u train_gars_stage2.py \
  model=univ2_decoder_h128 label=v3.12_h128_gc_focus \
  cache_path=/home/ubuntu/local_data/tls_patch_dataset_min4096.pt \
  train.class_weights=[1.0,5.0,5.0] train.gc_dice_weight=2.0 \
  > /tmp/v3.12.log 2>&1 &
V12_PID=$!
log "v3.12 PID=$V12_PID"
sleep 60
V12_DIR=$(ls -1dt $EXP/gars_stage2_v3.12_h128_gc_focus_* 2>/dev/null | head -1)
log "v3.12 dir = $V12_DIR"
wait_for_ckpt "$V12_DIR/best_checkpoint.pt"
wait_for_pid_dead "v3.12_h128_gc_focus"

# Step 11: cascade with v3.12 Stage 2 + v3.0 Stage 1.
run_eval v3.16_s1v3.0_s2v3.12 \
  $V0_CKPT $V12_DIR/best_checkpoint.pt \
  thresholds=[0.05,0.1,0.2,0.3,0.4,0.5] \
  min_component_size=2 closing_iters=1

log "Chain done."
