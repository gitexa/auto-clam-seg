"""Look for diff patches and artefacts that wandb_recover.py might have missed.

For each run we list every file (including hidden ones) and dump artefacts.
"""
from pathlib import Path

import wandb

PROJECT = "ajhaas/tls-pixel-seg"
OUT = Path("/home/ubuntu/auto-clam-seg/recovered")

RUNS = [
    "bmnj8pu1", "x50tcqt6", "so4ujuqa", "hwo25qzz",
    "mn5jorov", "cvrvn8yb", "ufz9a2o4", "qy3pj74h",
]

api = wandb.Api()
for run_id in RUNS:
    try:
        run = api.run(f"{PROJECT}/{run_id}")
    except Exception as e:
        print(f"  [skip] {run_id}: {e}")
        continue
    print(f"\n=== {run_id}  {run.name} ===")
    print("  files:")
    for f in run.files():
        print(f"    {f.name}  ({f.size} B)")
    print("  used artifacts (input):")
    try:
        for a in run.used_artifacts():
            print(f"    USED: {a.name}  type={a.type}")
    except Exception as e:
        print(f"    (err: {e})")
    print("  logged artifacts (output):")
    try:
        for a in run.logged_artifacts():
            print(f"    LOGGED: {a.name}  type={a.type}")
    except Exception as e:
        print(f"    (err: {e})")
