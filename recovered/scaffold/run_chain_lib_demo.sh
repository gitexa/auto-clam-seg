#!/usr/bin/env bash
# Demo / template chain script using the marker-based chain_lib.
# Used as the canonical pattern for new autoresearch chains so the
# pgrep-based wait fragility doesn't reappear.
#
# Pattern:
#   chain_run STEP1  /tmp/step1.log <command...>
#   chain_wait STEP1            # blocks until .done or .fail
#   chain_run STEP2  /tmp/step2.log <next command...>
#   chain_wait STEP2
#
# Each step's logs go to its own log file. Markers persist in
# $CHAIN_MARKER_DIR (default /tmp/chain_markers) so a chain can be
# inspected post-hoc or resumed.
set -uo pipefail
cd /home/ubuntu/auto-clam-seg/recovered/scaffold
source ./chain_lib.sh

EXP=/home/ubuntu/ahaas-persistent-std-tcga/experiments
log() { echo "[demo $(date +%H:%M:%S)] $*"; }

# Step 1: run a quick smoke (just to demonstrate the marker flow).
chain_run demo_step1 /tmp/demo_step1.log bash -c "echo step1; sleep 2; echo step1 done"
chain_wait demo_step1 || { log "step1 failed"; exit 1; }
log "step1 OK"

# Step 2: depends on step1 having finished.
chain_run demo_step2 /tmp/demo_step2.log bash -c "echo step2; sleep 2; echo step2 done"
chain_wait demo_step2 || { log "step2 failed"; exit 1; }
log "step2 OK"

log "demo chain complete"
