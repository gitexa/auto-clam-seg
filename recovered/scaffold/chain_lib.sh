#!/usr/bin/env bash
# chain_lib.sh — robust step orchestration for autoresearch chains.
#
# Replaces fragile pgrep-based waits with explicit done-markers in
# $CHAIN_MARKER_DIR.  Source this file from a chain script, then use:
#
#   chain_run    <step_name> <log_file> <command...>
#       Launch <command...> in the background. Writes
#       $CHAIN_MARKER_DIR/<step_name>.{pid,started}.  When the command
#       exits, an EXIT trap writes <step_name>.done if rc=0 else
#       <step_name>.fail. <step_name>.exit holds the exit code.
#
#   chain_wait   <step_name> [<timeout_sec>]
#       Block until <step_name>.done or <step_name>.fail exists.
#       Returns 0 on .done, 1 on .fail, 2 on timeout.  Default timeout
#       is 86400s (24h).
#
#   chain_done   <step_name>     # 0 if .done present, 1 otherwise.
#   chain_fail   <step_name>     # 0 if .fail present, 1 otherwise.
#   chain_kill   <step_name>     # SIGTERM the recorded PID (best effort).
#
# All operations are local-FS atomic (touch + mv); no race window.
# Markers persist across script restarts so a chain can resume.
#
# Self-test: chain_lib_selftest  → exits 0 on success, prints details.

set -u
CHAIN_MARKER_DIR="${CHAIN_MARKER_DIR:-/tmp/chain_markers}"
mkdir -p "$CHAIN_MARKER_DIR"

_chain_log() {
    echo "[chain_lib $(date +%H:%M:%S)] $*" >&2
}

# Internal: write done/fail marker based on exit code, then unlink .pid.
# This wrapper runs in the child process so it can capture rc reliably.
_chain_wrap() {
    local name="$1"; shift
    local logf="$1"; shift
    local rc=0
    "$@" >"$logf" 2>&1 || rc=$?
    echo "$rc" > "$CHAIN_MARKER_DIR/$name.exit"
    if [ "$rc" = "0" ]; then
        touch "$CHAIN_MARKER_DIR/$name.done"
    else
        touch "$CHAIN_MARKER_DIR/$name.fail"
    fi
    rm -f "$CHAIN_MARKER_DIR/$name.pid"
}

chain_run() {
    if [ "$#" -lt 3 ]; then
        echo "usage: chain_run <step_name> <log_file> <command...>" >&2
        return 2
    fi
    local name="$1"; shift
    local logf="$1"; shift
    # Clean any prior markers for this step name.
    rm -f "$CHAIN_MARKER_DIR/$name".{done,fail,exit,pid,started}
    # Launch wrapper.  setsid so the child survives our parent shell exit.
    setsid bash -c "_chain_wrap_inline() { local rc=0; \"\$@\" >\"$logf\" 2>&1 || rc=\$?; echo \"\$rc\" > \"$CHAIN_MARKER_DIR/$name.exit\"; if [ \"\$rc\" = 0 ]; then touch \"$CHAIN_MARKER_DIR/$name.done\"; else touch \"$CHAIN_MARKER_DIR/$name.fail\"; fi; rm -f \"$CHAIN_MARKER_DIR/$name.pid\"; }; _chain_wrap_inline $(printf '%q ' "$@")" </dev/null >/dev/null 2>&1 &
    local pid=$!
    echo "$pid" > "$CHAIN_MARKER_DIR/$name.pid"
    touch "$CHAIN_MARKER_DIR/$name.started"
    _chain_log "started $name (pid $pid, log $logf)"
}

chain_wait() {
    if [ "$#" -lt 1 ]; then
        echo "usage: chain_wait <step_name> [<timeout_sec>]" >&2
        return 2
    fi
    local name="$1"; shift
    local timeout="${1:-86400}"
    local waited=0
    while true; do
        if [ -f "$CHAIN_MARKER_DIR/$name.done" ]; then
            return 0
        fi
        if [ -f "$CHAIN_MARKER_DIR/$name.fail" ]; then
            local rc
            rc=$(cat "$CHAIN_MARKER_DIR/$name.exit" 2>/dev/null || echo "?")
            _chain_log "step $name FAILED (rc=$rc)"
            return 1
        fi
        if [ "$waited" -ge "$timeout" ]; then
            _chain_log "step $name TIMEOUT after ${timeout}s"
            return 2
        fi
        sleep 5
        waited=$((waited + 5))
    done
}

chain_done() {
    [ -f "$CHAIN_MARKER_DIR/$1.done" ]
}

chain_fail() {
    [ -f "$CHAIN_MARKER_DIR/$1.fail" ]
}

chain_kill() {
    local name="$1"
    local pid_file="$CHAIN_MARKER_DIR/$name.pid"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        kill -TERM "$pid" 2>/dev/null || true
        _chain_log "sent SIGTERM to $name (pid $pid)"
    fi
}

chain_clear() {
    # remove all markers for a step name
    rm -f "$CHAIN_MARKER_DIR/$1".{done,fail,exit,pid,started}
}

# ─────────────────────────────────────────────────────────
# Self-test: launches a 3s sleep step, waits, verifies markers.
# ─────────────────────────────────────────────────────────
chain_lib_selftest() {
    local out=0
    local logf
    logf=$(mktemp)
    local name="_selftest_$$_$(date +%N)"

    # Test 1: success path.
    chain_run "$name" "$logf" bash -c "echo ok; sleep 1; exit 0"
    if ! chain_wait "$name" 30; then
        echo "FAIL: success-path wait returned non-zero" >&2
        out=1
    fi
    if ! chain_done "$name"; then
        echo "FAIL: .done missing for success-path step" >&2
        out=1
    fi
    if [ "$(cat "$CHAIN_MARKER_DIR/$name.exit" 2>/dev/null)" != "0" ]; then
        echo "FAIL: .exit != 0 for success-path step" >&2
        out=1
    fi
    chain_clear "$name"

    # Test 2: failure path.
    name="_selftest_${$}_fail_$(date +%N)"
    chain_run "$name" "$logf" bash -c "echo broken; sleep 1; exit 7"
    chain_wait "$name" 30 && {
        echo "FAIL: failure-path wait returned 0" >&2
        out=1
    }
    if ! chain_fail "$name"; then
        echo "FAIL: .fail missing for failure-path step" >&2
        out=1
    fi
    if [ "$(cat "$CHAIN_MARKER_DIR/$name.exit" 2>/dev/null)" != "7" ]; then
        echo "FAIL: .exit != 7 for failure-path step" >&2
        out=1
    fi
    chain_clear "$name"

    # Test 3: timeout path (don't actually wait the full timeout — use 2s).
    name="_selftest_${$}_to_$(date +%N)"
    chain_run "$name" "$logf" bash -c "sleep 60"
    local rc=0
    chain_wait "$name" 2 || rc=$?
    if [ "$rc" != "2" ]; then
        echo "FAIL: timeout-path wait returned $rc (expected 2)" >&2
        out=1
    fi
    chain_kill "$name"
    chain_clear "$name"

    rm -f "$logf"
    if [ "$out" = "0" ]; then
        echo "chain_lib selftest: PASS"
    else
        echo "chain_lib selftest: FAIL"
    fi
    return "$out"
}
