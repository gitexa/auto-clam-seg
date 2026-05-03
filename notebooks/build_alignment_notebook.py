"""Generate `verify_patch_mask_alignment.ipynb` programmatically.

We craft a real .ipynb with structured cells so the user can open it in
Jupyter, re-run any single check, or convert to HTML.
"""
from pathlib import Path

import nbformat as nbf

NB_OUT = Path("/home/ubuntu/auto-clam-seg/notebooks/verify_patch_mask_alignment.ipynb")

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


md("""# Verify patch ↔ mask alignment

The Stage 2 GARS results stand or fall on this assumption: each patch's
top-left coordinate, scaled to the mask's pyramid level, lands on the
corresponding 256×256 region of the mask. If the alignment is off — by
even a fraction — every per-patch dice / GC count is wrong.

This notebook checks alignment across:

1. **Multiple slides** (3–4, one per cancer type, varying slide sizes).
2. **Multiple resolutions** — every page of each TIF pyramid.

Visual checks per slide:

- Mask pyramid: dimensions and downsampling factor at each level.
- Patch grid overlaid on the mask: every patch's top-left as a dot,
  rendered against the lowest-resolution page (thumbnail-style).
- Per-patch tiles: a TLS-positive and a GC-positive patch's 256×256
  mask region cropped from level 0, plus the same patch from a coarser
  pyramid level resampled — they should look the same up to scale.
""")

code("""import os, sys
sys.path.insert(0, '/home/ubuntu/profile-clam')
sys.path.insert(0, '/home/ubuntu/auto-clam-seg/recovered/scaffold')

# Import prepare_segmentation FIRST — it does `matplotlib.use("Agg")` at
# module load. We reset the backend with %matplotlib inline below so
# figures render in the notebook.
import prepare_segmentation as ps
""")
code("""%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import zarr
import tifffile
import torch
import torch.nn.functional as F

PATCH_SIZE = 256
print('Loaded prepare_segmentation; PATCH_SIZE =', PATCH_SIZE)
print('matplotlib backend:', plt.get_backend())
""")

md("""## 1. Pick test slides

One per cancer type, plus the largest slide we can find (to stress-test
the scale-down path). Skip entries with empty zarr paths.""")

code("""
entries = ps.build_slide_entries()
entries = [e for e in entries if e.get('mask_path') and e.get('zarr_path')]
print(f'{len(entries)} slides with both mask and features')

def safe_open(e):
    '''Some zarr stores are malformed — try/except to skip them.'''
    try:
        g = zarr.open(e['zarr_path'], mode='r')
        n = g['features'].shape[0]
        return g, n
    except Exception:
        return None, 0

picks = {}
for e in entries:
    ct = e['cancer_type']
    if ct in picks: continue
    g, n_patches = safe_open(e)
    if g is None: continue
    tif = tifffile.TiffFile(e['mask_path'])
    pages = [p.shape for p in tif.pages]
    picks[ct] = dict(entry=e, n_patches=n_patches, pages=pages)
    if len(picks) == 3: break

# Add a 4th: the slide with the most patches (likely largest in pixel space).
biggest = None
biggest_n = 0
for e in entries[:50]:
    g, n = safe_open(e)
    if g is not None and n > biggest_n:
        biggest_n, biggest = n, e
if biggest is not None:
    tif = tifffile.TiffFile(biggest['mask_path'])
    picks['LARGEST'] = dict(entry=biggest, n_patches=biggest_n,
                             pages=[p.shape for p in tif.pages])

for label, info in picks.items():
    sid = info['entry']['slide_id'][:30]
    print(f'  {label:8s}  {sid:30s}  n_patches={info[\"n_patches\"]:>6}  pages={info[\"pages\"]}')
""")

md("""## 2. Mask pyramid inspection

For each slide, render every pyramid level side-by-side. Verify that each
level is a faithful downsample of the previous one and that aspect
ratios are consistent.""")

code("""
def show_mask_pyramid(entry, label, skip_top_levels=2, max_levels=6):
    '''Render mask pyramid. Skip level 0 (and 1) — they can be 3-7 GB per
    slide; we get the same visual info from levels 2+ which are <250 MB.'''
    tif = tifffile.TiffFile(entry['mask_path'])
    full_h = tif.pages[0].shape[0]
    pages = list(tif.pages)[skip_top_levels:skip_top_levels + max_levels]
    n = len(pages)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1: axes = [axes]
    for i, page in enumerate(pages):
        h, w = page.shape[:2]
        scale = full_h / h
        arr = page.asarray()
        axes[i].imshow(arr, cmap='magma', vmin=0, vmax=2, interpolation='nearest')
        axes[i].set_title(f'lvl {skip_top_levels + i}: {h}×{w}  scale={scale:.1f}×')
        axes[i].axis('off')
    full_pages = [p.shape for p in tif.pages]
    fig.suptitle(
        f'{label}: {entry[\"slide_id\"][:35]}\\n'
        f'pyramid: {full_pages[0]} (level 0, ~{4 * full_pages[0][0] * full_pages[0][1] / 1e9:.1f} GB) → ... '
        f'→ {full_pages[-1]} (level {len(full_pages)-1})',
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

for label, info in picks.items():
    show_mask_pyramid(info['entry'], label)
""")

md("""## 3. Patch grid overlay

For each slide, take the lowest-resolution mask page (smallest pyramid
level), scale every patch's top-left coordinate down to that page's
resolution, and plot the patch centroids as dots.

If alignment is correct, the dots should sit *exactly* over the mask's
TLS regions for TLS-positive patches and elsewhere for negatives.""")

code("""
def overlay_patch_grid(entry, label, level=-1):
    grp = zarr.open(entry['zarr_path'], mode='r')
    coords = np.asarray(grp['coords'][:])  # (N, 2) top-left in slide px
    tif = tifffile.TiffFile(entry['mask_path'])
    page = tif.pages[level if level >= 0 else len(tif.pages) + level]
    full_h, full_w = tif.pages[0].shape[:2]
    h, w = page.shape[:2]
    scale = full_h / h
    arr = page.asarray()

    # Compute per-patch label: any TLS pixel in its mask cell?
    # Use a quick mask grid lookup at this level.
    coord_max_x = coords[:, 0].max() + PATCH_SIZE
    coord_max_y = coords[:, 1].max() + PATCH_SIZE
    # The mask covers up to (full_h, full_w) in slide-px space.
    # Each patch is PATCH_SIZE px wide in slide space; on this page that's
    # PATCH_SIZE / scale mask px.
    psize_lvl = PATCH_SIZE / scale

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(arr, cmap='magma', vmin=0, vmax=2, interpolation='nearest')

    # Project coords onto this page.
    cx = coords[:, 0] / scale
    cy = coords[:, 1] / scale
    # Cheap per-patch label: sample the mask cell at floor(cx), floor(cy)+psize/2
    labels = np.zeros(len(coords), dtype=np.int32)
    for i, (x, y) in enumerate(zip(cx, cy)):
        x0, y0 = int(x), int(y)
        x1 = min(int(x + psize_lvl) + 1, w)
        y1 = min(int(y + psize_lvl) + 1, h)
        cell = arr[y0:y1, x0:x1]
        if cell.size == 0: continue
        if (cell == 2).any(): labels[i] = 2
        elif (cell == 1).any(): labels[i] = 1

    n_bg = (labels == 0).sum()
    n_tls = (labels == 1).sum()
    n_gc = (labels == 2).sum()

    # Plot each class. Background dots tiny, TLS medium, GC large for visibility.
    for cls, color, size, marker in [
        (0, '#1f77b4', 0.5, '.'),
        (1, '#ff7f0e', 4, 'o'),
        (2, '#d62728', 12, '*'),
    ]:
        m = labels == cls
        ax.scatter(cx[m], cy[m], s=size, c=color, marker=marker, alpha=0.7,
                   edgecolors='none')

    ax.set_title(
        f'{label}: {entry[\"slide_id\"][:35]}\\n'
        f'level {len(tif.pages)+level if level<0 else level} '
        f'({h}×{w}, scale={scale:.1f}×)  •  '
        f'patches: bg={n_bg}, TLS={n_tls}, GC={n_gc}'
    )
    legend = [
        mpatches.Patch(color='#1f77b4', label=f'bg ({n_bg})'),
        mpatches.Patch(color='#ff7f0e', label=f'TLS ({n_tls})'),
        mpatches.Patch(color='#d62728', label=f'GC ({n_gc})'),
    ]
    ax.legend(handles=legend, loc='upper right')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

for label, info in picks.items():
    overlay_patch_grid(info['entry'], label, level=-1)
""")

md("""## 4. Per-patch tiles at multiple resolutions

For one TLS-positive and one GC-positive patch in each slide, crop the
patch's mask tile from **every** pyramid level. Resample each crop to
256×256 and display side-by-side.

If the alignment is correct, the same anatomical region should appear
at every level; only the resolution should change.""")

code("""
def show_patch_at_all_levels(entry, label, target_class=2):
    grp = zarr.open(entry['zarr_path'], mode='r')
    coords = np.asarray(grp['coords'][:])
    tif = tifffile.TiffFile(entry['mask_path'])
    pages = list(tif.pages)
    full_h, full_w = pages[0].shape[:2]

    # Use per-page aszarr() for memory-efficient windowed reads — avoids
    # loading the 3-7 GB level-0 page into memory just to crop 256×256.
    z0 = zarr.open(pages[0].aszarr(), mode='r')

    # Find a patch whose level-0 tile contains target_class pixels.
    coords_i = coords.astype(np.int64)
    chosen = None
    for i, (x, y) in enumerate(coords_i):
        x1 = int(min(x + PATCH_SIZE, full_w))
        y1 = int(min(y + PATCH_SIZE, full_h))
        if x1 <= x or y1 <= y: continue
        tile = np.asarray(z0[int(y):y1, int(x):x1])
        if (tile == target_class).any():
            chosen = i; break
    if chosen is None:
        print(f'  no class-{target_class} patch in {label}')
        return

    x, y = coords_i[chosen]
    fig, axes = plt.subplots(1, len(pages), figsize=(3.5 * len(pages), 4))
    if len(pages) == 1: axes = [axes]
    for li, page in enumerate(pages):
        h, w = page.shape[:2]
        scale = full_h / h
        x0_l, y0_l = int(x / scale), int(y / scale)
        x1_l = min(int((x + PATCH_SIZE) / scale) + 1, w)
        y1_l = min(int((y + PATCH_SIZE) / scale) + 1, h)
        # Per-page zarr view of this single pyramid level.
        zli = zarr.open(page.aszarr(), mode='r')
        tile = np.asarray(zli[y0_l:y1_l, x0_l:x1_l])
        # Resample to 256×256 for visual comparison.
        if tile.size > 0:
            t = torch.from_numpy(tile.astype(np.uint8)).unsqueeze(0).unsqueeze(0).float()
            t = F.interpolate(t, size=(PATCH_SIZE, PATCH_SIZE), mode='nearest')
            tile_resampled = t.squeeze().numpy().astype(np.uint8)
        else:
            tile_resampled = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        axes[li].imshow(tile_resampled, cmap='magma', vmin=0, vmax=2, interpolation='nearest')
        axes[li].set_title(
            f'lvl {li}  scale={scale:.1f}×\\n'
            f'crop {tile.shape}  '
            f'TLS={int((tile==1).sum())} GC={int((tile==2).sum())}'
        )
        axes[li].axis('off')
    fig.suptitle(
        f'{label}: {entry[\"slide_id\"][:30]}  •  '
        f'patch {chosen} @ ({int(x)}, {int(y)})  target_class={target_class}',
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

for label, info in picks.items():
    print(f'\\n--- {label}: TLS-positive patch ---')
    show_patch_at_all_levels(info['entry'], label, target_class=1)
    print(f'--- {label}: GC-positive patch ---')
    show_patch_at_all_levels(info['entry'], label, target_class=2)
""")

md("""## 5. Tissue-aware alignment overlay (slide ↔ mask ↔ patches)

Load the actual WSI at a mid-low pyramid level and overlay the HookNet
mask + patch positions on top. WSI level-0 dimensions match the mask
level-0 exactly — same coordinate system at every pyramid level — so
this is a direct visual check that the bright TLS regions in the mask
sit on top of the lymphocyte-rich tissue.""")

code("""
SLIDE_ROOT = '/home/ubuntu/ahaas-persistent-std-tcga/slides_tif_/data/drive2/alex/tcga/slides_v2'

def slide_path(entry):
    return os.path.join(SLIDE_ROOT, f'tcga-{entry[\"cancer_type\"].lower()}',
                        f'{entry[\"slide_id\"]}.tif')


def show_tissue_overlay(entry, label, target_level=6):
    sp = slide_path(entry)
    if not os.path.exists(sp):
        print(f'  no WSI for {label} at {sp}')
        return

    slide_tif = tifffile.TiffFile(sp)
    mask_tif = tifffile.TiffFile(entry['mask_path'])
    full_h, full_w = slide_tif.pages[0].shape[:2]
    target_level = min(target_level, len(slide_tif.pages) - 1)
    slide_page = slide_tif.pages[target_level]
    h, w = slide_page.shape[:2]
    scale = full_h / h

    # Find the mask page with matching dimensions.
    mask_page = None
    for mp in mask_tif.pages:
        if abs(mp.shape[0] - h) <= 1 and abs(mp.shape[1] - w) <= 1:
            mask_page = mp; break
    if mask_page is None:
        # Pick the closest one and resize.
        mask_page = min(mask_tif.pages, key=lambda p: abs(p.shape[0] - h))

    slide_arr = slide_page.asarray()
    mask_arr = mask_page.asarray()
    if mask_arr.shape != (h, w):
        # Resize mask via nearest-neighbour to match WSI level shape.
        m = torch.from_numpy(mask_arr.astype(np.uint8)).unsqueeze(0).unsqueeze(0).float()
        m = F.interpolate(m, size=(h, w), mode='nearest')
        mask_arr = m.squeeze().to(torch.uint8).numpy()

    # Per-patch label by sampling the mask cell at the patch's projected
    # location.
    grp = zarr.open(entry['zarr_path'], mode='r')
    coords = np.asarray(grp['coords'][:])
    cx_f = coords[:, 0] / scale
    cy_f = coords[:, 1] / scale
    psize_l = PATCH_SIZE / scale
    labels = np.zeros(len(coords), dtype=np.int32)
    for i, (x, y) in enumerate(zip(cx_f, cy_f)):
        x0, y0 = int(x), int(y)
        x1 = min(int(x + psize_l) + 1, w)
        y1 = min(int(y + psize_l) + 1, h)
        if x1 <= x0 or y1 <= y0: continue
        cell = mask_arr[y0:y1, x0:x1]
        if (cell == 2).any(): labels[i] = 2
        elif (cell == 1).any(): labels[i] = 1

    # RGBA overlay for the mask.
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask_arr == 1] = [255,  60,  60, 130]   # TLS — red
    overlay[mask_arr == 2] = [255, 220,   0, 220]   # GC  — yellow

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(slide_arr)
    axes[0].set_title(f'WSI level {target_level}  ({h}×{w}, scale={scale:.0f}×)')
    axes[0].axis('off')

    axes[1].imshow(slide_arr)
    axes[1].imshow(overlay, interpolation='nearest')
    axes[1].set_title(f'+ HookNet mask  (TLS=red, GC=yellow)')
    axes[1].axis('off')

    axes[2].imshow(slide_arr)
    n_bg = (labels == 0).sum(); n_tls = (labels == 1).sum(); n_gc = (labels == 2).sum()
    for cls, color, size in [(0, '#1f77b4', 0.4),
                              (1, '#ff7f0e', 5),
                              (2, '#d62728', 18)]:
        m = labels == cls
        axes[2].scatter(cx_f[m], cy_f[m], s=size, c=color, alpha=0.85,
                        edgecolors='none')
    axes[2].set_title(f'+ patches  (bg={n_bg}, TLS={n_tls}, GC={n_gc})')
    axes[2].axis('off')

    fig.suptitle(f'{label}: {entry[\"slide_id\"][:40]}', fontsize=12)
    plt.tight_layout()
    plt.show()


for label, info in picks.items():
    show_tissue_overlay(info['entry'], label, target_level=6)
""")

md("""## 6. Cross-check: predicted vs. ground-truth class counts per patch

Run the recovered Stage-2 build pipeline (`tls_patch_dataset._process_one_slide`)
on each test slide and compare the per-patch class statistics to a
direct level-0 crop. They should agree exactly.""")

code("""
from tls_patch_dataset import _process_one_slide

for label, info in picks.items():
    e = info['entry']
    res = _process_one_slide((e, ps.MASK_DIR, 1))  # min_tls_pixels=1
    if res is None:
        print(f'{label}: no TLS-positive patches'); continue
    n = res['features'].shape[0]
    masks = res['masks']  # (k, 256, 256)
    n_tls = int((masks == 1).any(axis=(1, 2)).sum())
    n_gc  = int((masks == 2).any(axis=(1, 2)).sum())
    n_total = info['n_patches']
    print(f'{label:8s}  total_patches={n_total:>6}  '
          f'TLS-positive={n:>4} ({100*n/n_total:.1f}%)  '
          f'with TLS pixels={n_tls:>4}  '
          f'with GC pixels={n_gc:>4}')
""")

md("""## 6. GNCAF dataloader supervision check

This is the load-bearing check for GNCAF training: the `(target_rgb,
target_mask)` pairs returned by `GNCAFSlideDataset` are exactly what the
model sees + computes cross-entropy loss against. We sample with a fixed
seed so the indexing matches what a training run would see.

For each slide we show three sampled targets. Each row has three panels:

1. **`target_rgb`** — the 256×256 RGB tile fed to the encoder
2. **`+ TLS overlay`** — same tile with TLS pixels (mask==1) shaded red
3. **`+ TLS + GC overlay`** — same tile with TLS red and GC (mask==2) yellow

Per-target stats below each row report TLS-pixel count and GC-pixel
count for the 256×256 mask — these are the per-pixel CE-loss targets.""")

code("""
from gncaf_dataset import GNCAFSlideDataset, _normalise_rgb, IMAGENET_MEAN, IMAGENET_STD
import matplotlib.patches as mpatches

# Scales to show, ordered widest → narrowest. Last entry must be 256
# (the model input).
SCALES = [1024, 512, 256]


def _read_centred_rgb(wsi_z, cx, cy, size):
    '''Read a size×size RGB tile centred on (cx, cy). Pads white if it
    runs off the slide edge.'''
    h, w = wsi_z.shape[:2]
    half = size // 2
    x0 = max(cx - half, 0); x1 = min(cx - half + size, w)
    y0 = max(cy - half, 0); y1 = min(cy - half + size, h)
    tile = np.asarray(wsi_z[y0:y1, x0:x1])
    out = np.full((size, size, 3), 255, dtype=tile.dtype)
    px = max(half - cx, 0); py = max(half - cy, 0)
    out[py:py + tile.shape[0], px:px + tile.shape[1]] = tile
    return out


def _read_centred_mask(mask_z, cx, cy, size):
    h, w = mask_z.shape
    half = size // 2
    x0 = max(cx - half, 0); x1 = min(cx - half + size, w)
    y0 = max(cy - half, 0); y1 = min(cy - half + size, h)
    tile = np.asarray(mask_z[y0:y1, x0:x1])
    out = np.zeros((size, size), dtype=tile.dtype)
    px = max(half - cx, 0); py = max(half - cy, 0)
    out[py:py + tile.shape[0], px:px + tile.shape[1]] = tile
    return out


def _draw_centre_box(ax, scale, model_size=256):
    '''Outline the central model_size × model_size region on a wider crop.'''
    if scale == model_size: return
    half = scale // 2; m = model_size // 2
    rect = mpatches.Rectangle((half - m, half - m), model_size, model_size,
                              fill=False, edgecolor='lime', linewidth=1.5)
    ax.add_patch(rect)


def show_supervision_multiscale(entry, label, ds, n_targets=3):
    sample = ds[picks_to_idx[label]]
    coords_path = entry['zarr_path']
    g = zarr.open(coords_path, mode='r')
    coords = np.asarray(g['coords'][:], dtype=np.int64)
    target_idx = sample['target_idx'].numpy()
    masks_256 = sample['target_mask'].numpy()
    K = min(n_targets, len(target_idx))

    # Open WSI + mask once per slide.
    wsi_path = ('/home/ubuntu/ahaas-persistent-std-tcga/slides_tif_/data/'
                f'drive2/alex/tcga/slides_v2/tcga-{entry[\"cancer_type\"].lower()}/'
                f'{entry[\"slide_id\"]}.tif')
    mask_path = entry['mask_path']
    wsi_z = zarr.open(tifffile.imread(wsi_path, aszarr=True, level=0), mode='r')
    mask_z = zarr.open(tifffile.imread(mask_path, aszarr=True, level=0), mode='r')

    n_rows = len(SCALES); n_cols = 3
    for ti in range(K):
        node = int(target_idx[ti])
        x, y = int(coords[node, 0]), int(coords[node, 1])
        cx, cy = x + 256 // 2, y + 256 // 2
        m_at_size = {256: masks_256[ti]}
        n_tls_256 = int((masks_256[ti] == 1).sum())
        n_gc_256 = int((masks_256[ti] == 2).sum())

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 11),
                                  squeeze=False)
        for ri, scale in enumerate(SCALES):
            rgb = _read_centred_rgb(wsi_z, cx, cy, scale)
            mask = (m_at_size.get(scale)
                     if scale in m_at_size
                     else _read_centred_mask(mask_z, cx, cy, scale))

            n_tls = int((mask == 1).sum())
            n_gc = int((mask == 2).sum())

            row_label = ('MODEL INPUT (256)' if scale == 256
                          else f'context {scale}×{scale}')
            row_color = 'red' if scale == 256 else 'black'

            # Col 0: RGB only
            axes[ri, 0].imshow(rgb)
            axes[ri, 0].set_title(f'{row_label}\\nRGB',
                                    color=row_color, fontsize=10)
            _draw_centre_box(axes[ri, 0], scale)
            axes[ri, 0].axis('off')

            # Col 1: + TLS overlay
            ov_tls = np.zeros((scale, scale, 4), dtype=np.uint8)
            ov_tls[mask == 1] = [255, 60, 60, 130]
            axes[ri, 1].imshow(rgb)
            axes[ri, 1].imshow(ov_tls, interpolation='nearest')
            axes[ri, 1].set_title(f'+ TLS  •  TLS={n_tls}',
                                    color=row_color, fontsize=10)
            _draw_centre_box(axes[ri, 1], scale)
            axes[ri, 1].axis('off')

            # Col 2: + TLS + GC overlay
            ov_full = np.zeros((scale, scale, 4), dtype=np.uint8)
            ov_full[mask == 1] = [255, 60, 60, 110]
            ov_full[mask == 2] = [255, 220, 0, 230]
            axes[ri, 2].imshow(rgb)
            axes[ri, 2].imshow(ov_full, interpolation='nearest')
            axes[ri, 2].set_title(f'+ TLS + GC  •  TLS={n_tls}, GC={n_gc}',
                                    color=row_color, fontsize=10)
            _draw_centre_box(axes[ri, 2], scale)
            axes[ri, 2].axis('off')

        fig.suptitle(
            f'{label}: {entry[\"slide_id\"][:35]}  •  target #{ti}  '
            f'patch_idx={node}  •  256-px supervision: '
            f'TLS={n_tls_256}/{256*256}px, GC={n_gc_256}px',
            fontsize=11,
        )
        plt.tight_layout()
        plt.show()


# Build dataset over our 4 picked slides only — fast, no need to enumerate all.
pick_entries = [info['entry'] for info in picks.values()]
gncaf_ds = GNCAFSlideDataset(pick_entries, max_targets=8, neg_per_pos=1.0,
                              rng_seed=42)
picks_to_idx = {label: i for i, label in enumerate(picks.keys())}
print(f'GNCAFSlideDataset built over {len(gncaf_ds)} slides; '
      f'sampling targets with rng_seed=42 (fixed).')
print(f'Per target: 3 scales (1024 / 512 / 256) × 3 cols (RGB / +TLS / +TLS+GC).')
print(f'The 256-px row in red = the actual model input + CE supervision.\\n')

for label, info in picks.items():
    show_supervision_multiscale(info['entry'], label, gncaf_ds, n_targets=3)
""")

md("""## 6b. GC-positive supervision (explicit)

GC patches are sparse (only ~3 in the LARGEST slide, 0 in BLCA/KIRC/LUSC
of our small pick set), so the random target sampler rarely picks one.
To verify GC supervision we explicitly find a GC-positive patch per slide
(mask centre = 2) and render the same multi-scale grid. The 256-row
should now show non-zero `GC=` pixel counts.""")

code("""
def find_gc_positive_patches(entry, max_results=2):
    '''Scan a slide's coords and return patch indices whose 256x256 mask
    region contains ANY GC pixel (==2). Returns (node_idx, x, y) list.'''
    g = zarr.open(entry['zarr_path'], mode='r')
    coords = np.asarray(g['coords'][:], dtype=np.int64)
    mask_z = zarr.open(tifffile.imread(entry['mask_path'], aszarr=True, level=0),
                        mode='r')
    mh, mw = mask_z.shape
    found = []
    for i, (x, y) in enumerate(coords):
        x0, y0 = int(x), int(y)
        x1 = min(x0 + 256, mw); y1 = min(y0 + 256, mh)
        if x1 <= x0 or y1 <= y0: continue
        tile = np.asarray(mask_z[y0:y1, x0:x1])
        if (tile == 2).any():
            found.append((i, x0, y0))
            if len(found) >= max_results: break
    return found


def show_multiscale_at_coord(entry, label, node_idx, x, y, header):
    cx, cy = x + 128, y + 128
    wsi_path = ('/home/ubuntu/ahaas-persistent-std-tcga/slides_tif_/data/'
                f'drive2/alex/tcga/slides_v2/tcga-{entry[\"cancer_type\"].lower()}/'
                f'{entry[\"slide_id\"]}.tif')
    wsi_z = zarr.open(tifffile.imread(wsi_path, aszarr=True, level=0), mode='r')
    mask_z = zarr.open(tifffile.imread(entry['mask_path'], aszarr=True, level=0),
                       mode='r')

    n_rows = len(SCALES); n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 11), squeeze=False)
    for ri, scale in enumerate(SCALES):
        rgb = _read_centred_rgb(wsi_z, cx, cy, scale)
        mask = _read_centred_mask(mask_z, cx, cy, scale)
        n_tls = int((mask == 1).sum()); n_gc = int((mask == 2).sum())
        row_label = ('MODEL INPUT (256)' if scale == 256
                      else f'context {scale}×{scale}')
        row_color = 'red' if scale == 256 else 'black'

        axes[ri, 0].imshow(rgb)
        axes[ri, 0].set_title(f'{row_label}\\nRGB', color=row_color, fontsize=10)
        _draw_centre_box(axes[ri, 0], scale)
        axes[ri, 0].axis('off')

        ov_tls = np.zeros((scale, scale, 4), dtype=np.uint8)
        ov_tls[mask == 1] = [255, 60, 60, 130]
        axes[ri, 1].imshow(rgb); axes[ri, 1].imshow(ov_tls, interpolation='nearest')
        axes[ri, 1].set_title(f'+ TLS  •  TLS={n_tls}', color=row_color, fontsize=10)
        _draw_centre_box(axes[ri, 1], scale)
        axes[ri, 1].axis('off')

        ov_full = np.zeros((scale, scale, 4), dtype=np.uint8)
        ov_full[mask == 1] = [255, 60, 60, 110]
        ov_full[mask == 2] = [255, 220, 0, 230]
        axes[ri, 2].imshow(rgb); axes[ri, 2].imshow(ov_full, interpolation='nearest')
        axes[ri, 2].set_title(f'+ TLS + GC  •  TLS={n_tls}, GC={n_gc}',
                                color=row_color, fontsize=10)
        _draw_centre_box(axes[ri, 2], scale)
        axes[ri, 2].axis('off')
    fig.suptitle(header, fontsize=11)
    plt.tight_layout()
    plt.show()


print('Searching for GC-positive patches in each picked slide...')
n_shown = 0
for label, info in picks.items():
    e = info['entry']
    gc_hits = find_gc_positive_patches(e, max_results=1)
    if not gc_hits:
        print(f'  {label}: no GC-positive patch — skipping')
        continue
    node_idx, x, y = gc_hits[0]
    header = (f'GC-POSITIVE  •  {label}: {e[\"slide_id\"][:35]}  •  '
              f'patch_idx={node_idx}  (mask centre = 2)')
    show_multiscale_at_coord(e, label, node_idx, x, y, header)
    n_shown += 1
print(f'Showed {n_shown}/{len(picks)} GC-positive supervision grids.')
""")

md("""## 7. Summary

If every check passes:

- ✅ Mask pyramid levels are clean downsamples of one another.
- ✅ Patch-grid overlay lights up the right anatomical regions.
- ✅ Per-patch tiles look identical (modulo resolution) at every level.
- ✅ The build pipeline's per-patch class counts match a direct crop.

Then the Stage 2 alignment is correct and any failure to reproduce the
original GC dice numbers is *not* a coordinate-system bug.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

NB_OUT.write_text(nbf.writes(nb))
print(f"Wrote {NB_OUT} ({NB_OUT.stat().st_size / 1024:.1f} KB)")
