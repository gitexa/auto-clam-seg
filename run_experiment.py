#!/usr/bin/env python3
"""Wrapper script that runs segmentation experiments with start/status/kill workflow.

Usage:
    # 1. Launch (non-blocking)
    python run_experiment.py start

    # 2. Check progress
    python run_experiment.py status

    # 3. Kill and collect results
    python run_experiment.py kill
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CLAM_DIR = "/home/ubuntu/profile-clam"
WORKTREE_DIR = os.path.join(Path(__file__).parent, "clam-worktree")
TRAIN_SCRIPT = os.path.join(WORKTREE_DIR, "train_segmentation.py")
PYTHON = os.path.join(CLAM_DIR, ".venv/bin/python")
BASE_DIR = Path(__file__).parent
STDOUT_LOG = BASE_DIR / "run_stdout.log"
STDERR_LOG = BASE_DIR / "run_stderr.log"
PID_FILE = BASE_DIR / ".run_pid"

METRICS = {
    "primary": "val_dice",
    "secondary": "det_auc",
    "count": "count_sp",
    "direction": "maximize",
}


def cmd_start(args):
    """Launch training in background."""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"ERROR: Experiment already running (pid={pid}). Use 'kill' first.")
            sys.exit(1)
        except OSError:
            PID_FILE.unlink()

    # Clean shared memory
    for f in Path("/dev/shm").glob("torch_*"):
        f.unlink(missing_ok=True)

    cmd = [PYTHON, TRAIN_SCRIPT]
    with open(STDOUT_LOG, "w") as out, open(STDERR_LOG, "w") as err:
        proc = subprocess.Popen(
            cmd, stdout=out, stderr=err, cwd=str(WORKTREE_DIR),
            preexec_fn=os.setsid,
        )

    PID_FILE.write_text(str(proc.pid))
    print(f"Started training (pid={proc.pid})")
    print(f"  stdout: {STDOUT_LOG}")
    print(f"  stderr: {STDERR_LOG}")
    print(f"  Monitor: python run_experiment.py status")


def cmd_status(args):
    """Check progress by reading EPOCH lines from stdout."""
    if not STDOUT_LOG.exists():
        print("No run log found. Start an experiment first.")
        return

    # Check if process is alive
    running = False
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            running = True
        except OSError:
            pass

    # Parse EPOCH lines
    epochs = []
    result_line = None
    with open(STDOUT_LOG) as f:
        for line in f:
            line = line.strip()
            if line.startswith("EPOCH "):
                parts = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
                epochs.append(parts)
            elif line.startswith("RESULT "):
                result_line = line

    status = "running" if running else "finished"
    print(f"Status: {status}")
    print(f"Epochs completed: {len(epochs)}")

    if epochs:
        last = epochs[-1]
        best_dice = max(float(e.get("val_dice", 0)) for e in epochs)
        print(f"\nLatest (epoch {last.get('epoch', '?')}):")
        for key in ["val_dice", "neg_dice", "count_sp", "count_r2", "det_auc",
                     "det_bacc", "bkt_bacc", "cls0_rec", "l_focal", "l_dice",
                     "l_center", "l_offset", "lr"]:
            if key in last:
                print(f"  {key}: {last[key]}")
        print(f"\nBest val_dice: {best_dice:.4f}")

        # Find best epoch
        best_ep = max(epochs, key=lambda e: float(e.get("val_dice", 0)))
        print(f"  at epoch {best_ep.get('epoch', '?')} "
              f"(count_sp={best_ep.get('count_sp', '?')}, "
              f"det_auc={best_ep.get('det_auc', '?')})")

    if result_line:
        print(f"\nFinal: {result_line}")

    # Check for errors
    if STDERR_LOG.exists():
        stderr = STDERR_LOG.read_text()
        errors = [l for l in stderr.split("\n") if "Error" in l or "error" in l.lower()]
        if errors:
            print(f"\nErrors found ({len(errors)}):")
            for e in errors[-3:]:
                print(f"  {e[:120]}")


def cmd_kill(args):
    """Kill running experiment and report results."""
    if not PID_FILE.exists():
        print("No running experiment found.")
        cmd_status(args)
        return

    pid = int(PID_FILE.read_text().strip())
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(2)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass
        print(f"Killed process group (pid={pid})")
    except OSError:
        print(f"Process {pid} already dead")

    PID_FILE.unlink(missing_ok=True)

    # Clean shared memory
    for f in Path("/dev/shm").glob("torch_*"):
        f.unlink(missing_ok=True)

    cmd_status(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("start")
    sub.add_parser("status")
    sub.add_parser("kill")
    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "kill":
        cmd_kill(args)
    else:
        parser.print_help()
