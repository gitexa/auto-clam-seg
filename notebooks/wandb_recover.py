"""Pull source-code artefacts and configs for the GARS runs from wandb.

Run after exporting WANDB_API_KEY (or after `wandb login`).

For each run we save:
  - config.yaml           (resolved config)
  - output.log            (stdout from the run)
  - code/<files>          (source files captured by wandb's code-save)
  - files/<files>         (any other run files)
into /home/ubuntu/auto-clam-seg/recovered/<run_id>/.
"""
from pathlib import Path

import wandb

PROJECT = "ajhaas/tls-pixel-seg"
OUT = Path("/home/ubuntu/auto-clam-seg/recovered")

RUNS = {
    "bmnj8pu1": "stage1_gcn3hop",
    "x50tcqt6": "stage1_gatv2_3hop",
    "so4ujuqa": "stage1_no_gnn",
    "hwo25qzz": "stage1_gcn3hop_512px",
    "mn5jorov": "stage2_univ2_decoder",
    "cvrvn8yb": "stage2_univ2_hidden128",
    "ufz9a2o4": "e2e_joint",
    "qy3pj74h": "unified_gatv2_decoder",
}

api = wandb.Api()
print(f"Authenticated as: {api.viewer.username}")

for run_id, label in RUNS.items():
    target = OUT / f"{run_id}_{label}"
    target.mkdir(parents=True, exist_ok=True)
    try:
        run = api.run(f"{PROJECT}/{run_id}")
    except Exception as e:
        print(f"  [skip] {run_id}: {e}")
        continue
    print(f"\n=== {run_id}  {run.name}  ({run.state}) ===")
    print(f"  → {target}")

    (target / "config.yaml").write_text(
        "\n".join(f"{k}: {v!r}" for k, v in run.config.items())
    )
    (target / "summary.txt").write_text(
        "\n".join(f"{k}: {v!r}" for k, v in run.summary.items())
    )

    for f in run.files():
        try:
            f.download(root=str(target), exist_ok=True)
            print(f"  ok  {f.name}  ({f.size} B)")
        except Exception as e:
            print(f"  err {f.name}: {e}")

print("\nDone.")
