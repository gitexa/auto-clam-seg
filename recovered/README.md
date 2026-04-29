# GARS run recovery — summary

Pulled on 2026-04-29 from wandb project `ajhaas/tls-pixel-seg` after the VM
crash that wiped `/home/ubuntu/autoresearch-clam-seg/clam-worktree/` and the
local-only branch `autoresearch/gars-cascade`.

## What was recovered

For each run, a directory `<run_id>_<label>/` containing:

| File | Content |
|---|---|
| `config.yaml` | Full resolved hyperparameter config |
| `output.log` | Stdout — per-epoch metrics + crash traces |
| `wandb-summary.json` | Final / best metrics |
| `wandb-metadata.json` | Git commit hash, entry-point file, Python/CUDA env |
| `requirements.txt` | Pinned package versions at run time (4 413 B) |

| Run ID | Label | State | Commit | Entry point |
|---|---|---|---|---|
| `bmnj8pu1` | stage1_gcn3hop | crashed | (not in metadata) | (no codePath) |
| `x50tcqt6` | stage1_gatv2_3hop | finished | `100efb2a` | (no codePath) |
| `so4ujuqa` | stage1_no_gnn | finished | (not recorded) | — |
| `hwo25qzz` | stage1_gcn3hop_512px | finished | (not recorded) | — |
| `mn5jorov` | stage2_univ2_decoder | finished | `be3a7b60` | `train_gars_stage2.py` |
| `cvrvn8yb` | stage2_univ2_hidden128 | finished | (not recorded) | — |
| `ufz9a2o4` | e2e_joint | finished | (not recorded) | — |
| `qy3pj74h` | unified_gatv2_decoder | crashed | `983fc227` | `train_gars_unified.py` |

## What was NOT recovered

**Source code — confirmed gone.** None of the runs had
`wandb.run.log_code()` enabled, so no `code/` files, no `diff.patch`, and no
logged artefacts. The git commits above point at
`https://github.com/gitexa/profile-clam.git`, but on 2026-04-29 with
authenticated `git fetch origin <sha>` GitHub answered **"not our ref"** for
all three:

```
fatal: remote error: upload-pack: not our ref 100efb2a7f1c18e7dd475c8f0053835f2929157b
fatal: remote error: upload-pack: not our ref be3a7b6003ba52029dfd2c191b83e85b247460d0
fatal: remote error: upload-pack: not our ref 983fc227139fdcde7a4e1a86d67d2f292564c87d
```

`git ls-remote origin` shows only `main` (`6a452f8`) and
`autoresearch/seg-apr11` (`053b944`); the remote has no `gars-cascade`
branch and no orphan commits matching the wandb-recorded SHAs. The branch
was never pushed.

## How to revive the GARS pipeline from these artefacts

1. **`config.yaml` is enough to re-create the training entry-point**:
   the `output.log` documents every metric the run printed, and these are
   trivially mapped from the config keys.
2. **Architecture skeletons are in `wandb-metadata.json`**: file names tell
   you which scripts to write (`train_gars_stage2.py`, `train_gars_unified.py`)
   and the Stage 1 metric pattern (recall/precision/F1 + selected count) tells
   you it was a binary patch classifier.
3. **The persistent volume still holds `best_checkpoint.pt`** for every
   completed run at `/lambda/nfs/ahaas-persistent-std-tcga/experiments/`. So
   even without source you have:
    - The exact weights that produced these results
    - The exact hyperparameters that produced them
    - The pinned package versions to load them under
4. The `RECONSTRUCTION_apr13_apr27.md` notebook lists the observed
   architectures (Stage 1 = GraphTLSDetector ~626 K params; Stage 2 =
   UNIv2PixelDecoder ~9.3 M params; Unified = GraphEnrichedDecoder
   ~10.15 M params) which combined with the configs and the hooknet mask /
   UNI-v2 feature loading already in `prepare_segmentation.py` makes a clean
   re-implementation tractable.

## To recover the actual source (optional, requires GitHub auth)

```bash
# in /home/ubuntu/profile-clam, with gh CLI authenticated or
# GITHUB_TOKEN set in env:
git fetch origin 100efb2a7f1c18e7dd475c8f0053835f2929157b   # stage1
git fetch origin be3a7b6003ba52029dfd2c191b83e85b247460d0   # stage2
git fetch origin 983fc227139fdcde7a4e1a86d67d2f292564c87d   # unified
git checkout -b gars-recovered be3a7b60
ls train_gars_*.py
```

If those commits aren't on the remote, they're gone — re-implement from
`config.yaml` + the architecture descriptions in
`auto-clam-seg/notebooks/RECONSTRUCTION_apr13_apr27.md`.

## To prevent this next time

Add to `train_*.py` after `wandb.init(...)`:

```python
wandb.run.log_code(".", include_fn=lambda p: p.endswith(".py"))
```

Or set the env var globally:

```bash
export WANDB_DISABLE_CODE=false
```

Wandb will then upload your training source alongside every run.
