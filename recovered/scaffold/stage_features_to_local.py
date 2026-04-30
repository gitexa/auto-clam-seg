"""Stage UNI-v2 zarr feature stores from NFS to local SSD.

Each zarr is a directory tree (zarr v3 sharded). We copy in parallel
using rsync so partial copies can be resumed. Idempotent: skips zarrs
whose local copy has the same total size as NFS source.

After staging, training code can use local copies by setting
prepare_segmentation.ZARR_DIRS to the local paths (see `local_zarr_dirs()`
or pass `--use_local_ssd` to train_gars_stage1.py).

Run as:
    python stage_features_to_local.py [--n_workers 8] [--dry_run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/profile-clam")
import prepare_segmentation as ps  # noqa: E402

LOCAL_ROOT = Path("/home/ubuntu/local_data/zarr")


def local_zarr_dirs() -> dict[str, str]:
    """Mirror of `prepare_segmentation.ZARR_DIRS` rooted at local SSD."""
    return {ct: str(LOCAL_ROOT / ct) for ct in ps.ZARR_DIRS}


def _dir_size(path: Path) -> int:
    """Total bytes in a directory tree (small zarrs ~1k files each)."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _copy_one(args: tuple[Path, Path]) -> dict:
    src, dst = args
    name = src.name
    src_size = _dir_size(src)
    if dst.exists():
        dst_size = _dir_size(dst)
        if dst_size == src_size:
            return {"name": name, "status": "skip", "bytes": src_size}
        # Partial / mismatched: nuke and recopy.
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # rsync handles the directory tree atomically and is fast for many
    # small files.
    rc = subprocess.call([
        "rsync", "-a", "--no-perms", "--no-owner", "--no-group",
        f"{src}/", f"{dst}/",
    ])
    if rc != 0:
        return {"name": name, "status": "error", "bytes": 0}
    return {"name": name, "status": "copy", "bytes": src_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only stage the first N zarrs per cancer type "
                         "(useful for smoke tests)")
    args = ap.parse_args()

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []
    total_src = 0
    for ct, src_dir in ps.ZARR_DIRS.items():
        local_ct = LOCAL_ROOT / ct
        local_ct.mkdir(parents=True, exist_ok=True)
        zarrs = sorted(Path(src_dir).glob("*_complete.zarr"))
        if args.limit:
            zarrs = zarrs[: args.limit]
        for z in zarrs:
            sz = _dir_size(z)
            total_src += sz
            pairs.append((z, local_ct / z.name))
        print(f"  {ct}: {len(zarrs)} zarrs at {src_dir}")

    print(f"\nTotal: {len(pairs)} zarrs, {total_src / 1e9:.1f} GB on NFS")
    free = shutil.disk_usage("/home/ubuntu/local_data").free
    print(f"Local SSD free: {free / 1e9:.1f} GB")
    if total_src > free * 0.95:
        print(f"  WARNING: source is > 95% of free space — abort")
        return
    if args.dry_run:
        print("(dry run)")
        return

    print(f"\nCopying {len(pairs)} zarrs with {args.n_workers} workers...")
    t0 = time.time()
    n_copy = n_skip = n_err = 0
    bytes_done = 0
    bytes_target = sum(_dir_size(s) for s, _ in pairs)
    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = [pool.submit(_copy_one, p) for p in pairs]
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res["status"] == "copy":
                n_copy += 1; bytes_done += res["bytes"]
            elif res["status"] == "skip":
                n_skip += 1; bytes_done += res["bytes"]
            else:
                n_err += 1
            if done % 50 == 0:
                el = time.time() - t0
                gbps = bytes_done / max(1, el) / 1e9
                print(f"  [{done:>4}/{len(pairs)}]  copied={n_copy} "
                      f"skipped={n_skip} err={n_err}  "
                      f"{bytes_done/1e9:.1f}/{bytes_target/1e9:.1f} GB  "
                      f"({gbps:.2f} GB/s)")

    el = time.time() - t0
    print(f"\nDone in {el:.0f}s  (copied={n_copy}, skipped={n_skip}, errored={n_err})")
    print(f"Local zarr root: {LOCAL_ROOT}")
    print(f"Use via: ps.ZARR_DIRS = {local_zarr_dirs()}")


if __name__ == "__main__":
    main()
