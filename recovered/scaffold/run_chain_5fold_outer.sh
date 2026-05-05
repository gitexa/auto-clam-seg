#!/usr/bin/env bash
# Outer 5-fold orchestration: runs v3.58 → cascade → v3.56 sequentially.
# Each inner chain is itself a chain_lib chain over folds 1..4.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold
source ./chain_lib.sh

log() { echo "[outer 5fold $(date +%H:%M:%S)] $*"; }

# v3.58 (12L+aug, paper-faithful) — already running; just wait for it
log "Phase 1: waiting for v3.58 5-fold chain to complete"
while pgrep -f run_chain_5fold_gncaf_v358 >/dev/null 2>&1; do
  sleep 60
done
log "Phase 1 done"

# Cascade (Stage 1 + Stage 2)
log "Phase 2: launching cascade 5-fold chain"
chain_run cascade_5fold_outer /tmp/cascade_5fold_outer.log \
  bash ./run_chain_5fold_cascade.sh
chain_wait cascade_5fold_outer || log "cascade 5fold chain failed"
log "Phase 2 done"

# GNCAF v3.56 (6L unfrozen, deployment variant)
log "Phase 3: launching GNCAF v3.56 5-fold chain"
chain_run v356_5fold_outer /tmp/v356_5fold_outer.log \
  bash ./run_chain_5fold_gncaf_v356.sh
chain_wait v356_5fold_outer || log "v3.56 5fold chain failed"
log "Phase 3 done"

log "===== ALL 5-FOLD CHAINS COMPLETE ====="
