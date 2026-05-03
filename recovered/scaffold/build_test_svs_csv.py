"""Build a CSV of test-slide SVS paths for save_image_at_spacing.py."""
from __future__ import annotations
import sys, os
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/ubuntu/profile-clam")
import prepare_segmentation as ps

ps.set_seed(42)
entries = ps.build_slide_entries()
folds_pair, test_entries = ps.create_splits(entries, k_folds=1, seed=42)
test_mask = [e for e in test_entries if e.get("mask_path")]

SVS_ROOT = "/lambda/nfs/ahaas-persistent-std-tcga/slides"

rows = []
missing = []
for e in test_mask:
    cancer = e["cancer_type"].lower()
    sid = e["slide_id"]
    svs = Path(SVS_ROOT) / f"tcga-{cancer}" / "slides" / f"{sid}.svs"
    if svs.exists():
        rows.append({"slide_id": sid, "cancer_type": e["cancer_type"], "slide_path": str(svs)})
    else:
        missing.append((sid, str(svs)))

df = pd.DataFrame(rows)
out = Path("/tmp/test_slides_svs.csv")
df.to_csv(out, index=False)
print(f"Wrote {out}: {len(df)} test slides with SVS available")
print(f"Missing SVS: {len(missing)}")
for sid, p in missing[:5]:
    print(f"  {sid} -> {p}")
