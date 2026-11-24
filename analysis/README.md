# analysis/

Scripts that turn the result files into the paper. None of them runs a clustering
experiment except `controls_v2.py --check-spectral` and `make_pipeline_figure.py`,
which regenerate a handful of datasets.

| script | writes | paper |
|---|---|---|
| `audit_paper_claims_v2.py` | console report; exit status 1 on any mismatch | every number quoted in the text (267 checks) |
| `make_paper_tables_v2.py` | `tables_paper/tab_*.tex` | Tables 2, 4–8 and S1–S20 |
| `make_paper_figures_v2.py` | `figures_paper/*.pdf, *.png` | Figures 4–11 and S1–S3 |
| `make_pipeline_figure.py` | `figures_paper/pipeline_stages.*` | Figure 3 |
| `make_factors_figure.py` | `figures/factors_2d.*` | Figure 2 |
| `audit_value_v2.py` | `tab_threshold_conventions`, `tab_paired_effects`, `tab_audit_predictive`; `predictive_summary.csv`, `paired_ratios.csv` (read by the claims audit) | S6.6–S6.8, §6.4 |
| `controls_v2.py` | `tab_refinement_control`, `tab_K_control` | S6.10, S6.11 |
| `controls_v2.py --check-spectral` | `benchmarks/classical/results/sweeps_v2/spectral_imbalance_check.csv` | S7.4, Table 8 |
| `q1_ablation.py` | `tables/q1_*.csv` | Section 6.1 (stand-alone version) |
| `q2_instrument_sensitivity.py` | `tables/q2_instrument_sensitivity.csv` | Section 6.2, Table S5 (needs `datasets/suite`) |

All scripts take `--results-root` (default: repository root) and, where they write,
`--out`.

`reference_tables/` holds the tables of the paper as submitted; `scripts/reproduce.py
verify-release` regenerates every table and requires it to be byte-identical to them.

Conventions shared by the scripts: the suite pool of the paper is the nine classical
methods (DPC is reported separately); a run with `run_status =
no_admissible_configuration` (protocols S and D) is a degenerate output and counts as a
detectable failure with inlier ARI 0; a run with `failure_mode = error` would be
excluded from the aggregates and reported (the shipped results contain none).
