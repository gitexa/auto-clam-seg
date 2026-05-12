# Rigorous evaluation summary — fold-0 fullcohort

All numbers from per-slide rows under the union TLS Dice semantic (GC ⊂ TLS). Bootstrap 95% CI on N=124 positive slides.

| Architecture | mDice | TLS Dice [95% CI] | GC Dice [95% CI] | TLS-FP | GC-FP | TLS det F1 | GC det F1 | TLS Spearman | GC Spearman |
|---|---|---|---|---|---|---|---|---|---|
| Cascade v3.37 | 0.719 | 0.606 [0.567, 0.640] | 0.831 [0.782, 0.877] | 41.5% | 4.9% | 0.924 | 0.894 | 0.911 | 0.907 |
| GNCAF v3.58 | 0.432 | 0.276 [0.242, 0.308] | 0.587 [0.508, 0.665] | 95.1% | 12.2% | 0.864 | 0.652 | 0.550 | 0.595 |
| GNCAF v3.62 (paper-strict) | 0.508 | 0.331 [0.297, 0.364] | 0.685 [0.602, 0.767] | 85.4% | 2.4% | 0.876 | 0.133 | 0.715 | 0.124 |
| GNCAF v3.63 (dual-σ, heavy) | 0.435 | 0.200 [0.175, 0.225] | 0.669 [0.590, 0.745] | 97.6% | 12.2% | 0.861 | 0.684 | 0.648 | 0.623 |
| GNCAF v3.65 (dual-σ, simple) | 0.469 | 0.275 [0.242, 0.308] | 0.664 [0.591, 0.735] | 92.7% | 4.9% | 0.867 | 0.750 | 0.708 | 0.727 |
| GNCAF v3.65 + Stage 1 gate | 0.472 | 0.272 [0.239, 0.305] | 0.672 [0.600, 0.743] | 41.5% | 4.9% | 0.924 | 0.750 | 0.776 | 0.727 |
| seg_v2.0 (tls_only) | 0.477 | 0.592 [0.560, 0.621] | 0.362 [0.283, 0.438] | 31.0% | 0.0% | 0.938 | 0.667 | 0.918 | 0.640 |
| seg_v2.0 (dual) | 0.667 | 0.591 [0.561, 0.620] | 0.742 [0.673, 0.805] | 61.9% | 2.4% | 0.893 | 0.775 | 0.881 | 0.764 |
| seg_v2.0 (dual) + Stage 1 gate | 0.661 | 0.580 [0.543, 0.612] | 0.742 [0.673, 0.805] | 40.5% | 2.4% | 0.911 | 0.775 | 0.892 | 0.764 |
| Cascade v3.38 (dual-σ Stage 2) | 0.703 | 0.607 [0.568, 0.641] | 0.798 [0.737, 0.851] | 41.5% | 4.9% | 0.924 | 0.851 | 0.911 | 0.867 |

## Definitions

- **TLS Dice (union)**: target_tls = (gt ≥ 1) — GC pixels count as TLS biologically. Per-slide mean over positives.
- **TLS-FP rate**: fraction of GT-negative slides where `n_tls_pred > 0` after post-processing (min_size=1, closing_iters=0).
- **TLS det F1**: slide-level binary F1 — positive = (gt_n_tls > 0), pred-positive = (n_tls_pred > 0).
- **TLS Spearman**: rank correlation between `n_tls_pred` and metadata gold-standard `gt_n_tls` over all slides.