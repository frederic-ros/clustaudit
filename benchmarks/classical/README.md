# benchmarks/classical/

Classical tier: K-Means, GMM, Spectral, Ward, Average, Complete, DBSCAN, HDBSCAN,
OPTICS and density peaks (DPC).

| script | what | results |
|---|---|---|
| `exp1_difficulty_validation.py` | the 75-dataset suite (3 regimes × 5 variants × 5 seeds) under `--criterion silhouette \| dbcv \| fixed`; controls `--no-prerefine`, `--refine-classifier`, `--regime`, `--algos` | `results/exp1_difficulty_validation/` |
| `exp2_sensitivity_analysis.py` | noise grid (amount × placement), anisotropy, dimension (`--fixed-n`) | `results/exp2_sensitivity_analysis/` |
| `exp6_calibrated_generation.py` | parametric vs calibrated generation on measured targets (Section 6.6) | `results/exp6_calibrated_generation/` |
| `../../scripts/sweeps_v2.py` | single-factor sweeps (overlap, imbalance, densities, connectivity, `eta_int`); `--K`, `--n`, `--no-prerefine` | `results/sweeps_v2/` |
| `tune_algo.py` | the tuning protocols used by exp1/exp2 (thin wrapper over `clustaudit.audit.algorithms`) | — |

Tuning protocols: (S) per-dataset grid search on the Silhouette; (D) as (S) but DBCV
for the density methods (needs `hdbscan`); (F) fixed defaults, true K given to the
K-type methods.
