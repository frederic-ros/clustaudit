# Changelog

## 2.2.1 — 2026-11-21

The version described in the paper. Two fixes, and no result changes: the result files,
the suite, every table and every number of the paper are those of 2.2.0.

- `analysis/make_paper_figures_v2.py`: figures readable in monochrome print and at the
  text width. Every method has its own outlined marker, and the methods of a family
  differ by line style (solid, dashed, dotted, dash-dot); profile panels and the noise
  grid use the same marker/line-style cycle; curve figures are drawn at about twice their
  printed width with larger fonts, legends under the figures. No number changes: the
  verification (`reproduce.py verify-release`) passes unchanged (243/243 claims,
  25/25 tables).
- `scripts/reproduce.py`: `verify-release` (and `from-scratch`) failed on Windows. The
  scripts print non-ASCII text, and their output, redirected to log files, was encoded
  in the Windows code page (cp1252): the claims audit and the pipeline figure stopped
  with `UnicodeEncodeError`. Every step now runs with UTF-8 output (`PYTHONUTF8=1`),
  and logs and reports are read and written in UTF-8. No result changes.

## 2.2.0 — first public release

### No run is ever lost (affects protocols S and D)
- Before this release, a tuner whose grid contained no admissible configuration returned
  `None` and the run was silently dropped, and a method that raised was skipped (bare
  `except: pass` in exp2). Under protocol (S), DBSCAN was thus missing from 21 of the 75
  suite datasets, from 86 of the 225 noise-grid cells and from part of the other grids;
  protocol (F) was not affected. Tuners now return `NoAdmissibleConfiguration` (no
  partition: a degenerate output, i.e. a detectable failure, `run_status =
  no_admissible_configuration`); a method that raises gives an error row
  (`failure_mode = error`, message in `run_error`). `audit_dataset`, `run_pool`
  (`with_status=True`), `run_method`, the experiment scripts and the real-data transfer
  all record every run. The (S) and (D) results were regenerated.
- The overlap sweep under (S) had two seeds; it now has five, like every sweep.

### Verification and reproduction
- `scripts/reproduce.py`: one declaration of every step (command, output files, expected
  rows) for all platforms, and two modes. `verify-release` checks the distributed
  results (tests; completeness and SHA-256 of every result file against
  `results_manifest.csv`; every number of the paper; every table regenerated
  byte-identical to `analysis/reference_tables/`) and says that it re-runs nothing.
  `from-scratch` re-runs the experiments in a new directory, checks each step's outputs
  (written by the step, expected rows), stops at the first failure, reports skipped steps
  as NOT RUN and writes `REPRODUCTION_REPORT.md`. `run_all_v2.sh/.bat` and
  `verify_release.sh/.bat` are thin wrappers; the former run-all scripts went on after a
  failed step and could end with passing checks on stale result files.
- Found by declaring the outputs: two exp2 runs wrote files under names the analyses did
  not read (`exp2 --out` added); two files read by the analyses (`frac_relabelled_knn*.csv`)
  were written by no script (the fractions are now read from the `frac_relabelled` column
  of exp1); the DPC-only files are no longer needed (DPC is in every pool).
- Two shipped files had been produced by an older version of the generator (the
  anisotropy/dimension grid and the fixed-n dimension grid under (S)); the from-scratch run
  replaced them. The distributed classical and cross-generator results (34 of the 40
  declared files) were all produced by `from-scratch` runs in the pinned environment; the
  deep-tier and real-data files come from earlier runs of the same scripts.
- The analyses parse result files exactly (`float_precision="round_trip"`): the default
  parser of some pandas versions returns an adjacent double, which moved one printed
  value across a rounding boundary. Summary files are written with ten significant
  digits and compared numerically. A constant column prints `--`, never `+nan`.
- The claims audit declares how many claims it evaluates (243): a claim whose input file
  is missing is a failure, not a smaller denominator.
- Integration tests (`tests/test_integration.py`) and a GitHub Actions workflow
  (`.github/workflows/verify.yml`): tests and claims on Python 3.9-3.12 with the version
  ranges, `verify-release` in the pinned environment, and an on-demand from-scratch job.

### Earlier in 2.2.0

- Repository aligned with the paper: scripts that only served the first version of
  the paper were removed; READMEs rewritten in English; comments of the deep and
  real-data scripts translated.
- `run_transfer.py --relabel` writes `transfer_*_relabel.csv` instead of overwriting
  the labels-as-given results.
- The overlap and density sweeps under protocol (F) were re-run with the current
  evaluation (smallest-cluster F1, failure mode columns); inlier ARIs are unchanged.
- The overlap / density sweeps now include DPC; the analysis scripts append the older
  DPC-only files only when DPC is missing (no duplicated rows). The protocol-(S) panel of
  the overlap figure uses DPC's own Silhouette run.
- `audit_value_v2.py` also writes `sweeps_v2/paired_ratios.csv` and the upper bounds of
  the bootstrap intervals; `audit_paper_claims_v2.py` checks 179 numbers, the predictive
  ones at half a unit of the last printed digit.
- Python 3.9-3.11 compatibility (no backslash inside f-string expressions); verified from
  a clean copy on 3.9, 3.10, 3.11 and 3.12 (tests, claims, byte-identical tables).
- `run_all_v2.bat`: two leftover lines in bash syntax removed (K = 3 / 8 controls are run
  once each); `q2_instrument_sensitivity.py --help` no longer crashes. Every flag used by the
  reproduction scripts was checked against the scripts' parsers.
- Pre-generated suite (`datasets/suite`), LICENSE (MIT), CITATION.cff, .zenodo.json,
  pinned `requirements.txt`, `environment.yml`.

## 2.1.8 — calibrated generation, the value of the audit, controls

- Controls for the fragile points: `exp1_difficulty_validation.py --refine-classifier
  {knn,svm,random_forest} --regime {easy,medium,hard}`; `sweeps_v2.py --K` and
  `--no-prerefine`; `analysis/controls_v2.py` writes the refinement-classifier and
  number-of-clusters tables (S6.10, S6.11). Result files of these runs are shipped.
- `audit_value_v2.py`: per-generator gains (MDCGen / HAWKS) and a summary file
  `benchmarks/cross-generator/results/predictive_summary.csv` read by the claims audit.
- `controls_v2.py --check-spectral`: regenerates the imbalance sweep at rho = 5 and records
  what Spectral's cluster for the smallest reference cluster contains (explains ARI 1.00 /
  F1 0.50 in Table 8); result file shipped.
- `audit_paper_claims_v2.py`: relabel fractions, controls, predictive value.

- `clustaudit/generators/calibrate.py`: `calibrate(target, seed, ...)` searches
  (sigma, alpha, eta, eta_int) so that the *measured* profile reaches a target on
  chosen descriptors (derivative-free pattern search, <= `budget` evaluations).
- `benchmarks/classical/exp6_calibrated_generation.py`: parametric vs calibrated
  generation on 8 measured targets x 5 seeds, then the pool under (F) on both
  (Section 6.6 / Table 7 of the paper; ~10 min).
- `analysis/audit_value_v2.py`: (i) out-of-generator prediction of the ARI from the
  operating envelope vs leaderboard score vs Silhouette fit, with pairwise ordering
  accuracy; (ii) paired vs independent level-to-level effects; (iii) critical levels
  and failure modes under alternative cut-offs. Writes three supplement tables.
- `metricsstate`: the core Silhouette is NaN instead of raising when fewer than
  three cores remain (very heavy overlap); the profile then reports it as such.
- `scripts/sweeps_v2.py --no-prerefine`: raw generative labels as reference (control
  for the emblematic sweeps; results shipped: identical curves).
- `audit_value_v2.py`: bootstrap intervals over external datasets for the gains.
- `audit_paper_claims_v2.py`: 9 claims added (101 in all).

## 2.1.7 — paper tables and figures are generated

- `analysis/make_paper_tables_v2.py`: every table of the manuscript and of the
  supplement, written as LaTeX `tabular` fragments (`analysis/tables_paper/`)
  from the result CSVs (ablation under S/D/F, grouped cross-validation, scalar
  candidates across regimes, instrument sensitivity, divergence, failure modes,
  imbalance incl. NMI/recall and n = 3000, factor x descriptor matrix, critical
  levels with seed ranges, cross-generator, real data, deep tier). DPC rows
  (`*_DPC.csv`) are merged when present.
- `analysis/make_paper_figures_v2.py`: every figure of the manuscript and of
  the supplement (`analysis/figures_paper/`), one style for all of them
  (family colours, seed bands, shared legends). The real-data figure's second
  panel is now a sorted dot plot (the labels of the scatter were unreadable).
- `knn_profile`: `noise_proximity` is `0` on a noise-free dataset (was `NaN`,
  which silently dropped the five noise-free suite datasets from any regression
  that used it). The analysis scripts fill `NaN` with `0` for CSVs written
  before this version, so old result files give the same tables. Table 3 of the
  manuscript changes in its last three rows (e.g. connectivity family,
  components + k-NN: 0.79 -> 0.84); no conclusion changes.
- `analysis/make_pipeline_figure.py`: one 2-D dataset through the five stages
  (sampled labels, pre-noise refinement with its accuracy trajectory, noise as
  placed, STATE as measured), every number read from the generation trace.
- `evaluate_partition`: new column `min_cluster_f1_inlier` (precision over ground-truth
  inliers only). `min_cluster_f1` is unchanged (precision over every point of the
  returned cluster, noise included), so result files stay comparable; the new column
  separates absorption of a rare cluster from noise attached to it (Spectral on the
  imbalance sweep: inlier ARI 1.00, F1 0.50). Rerun the imbalance sweeps to fill it.
- Vocabulary: the failure mode stored as `declared` is called *detectable* in the
  paper (a degenerate output is detected by the protocol, not announced by the method).
- `make_paper_tables_v2.py` also writes `tab_audit_cards.tex`: the standardized
  output of the battery, one row per method (critical levels with seed ranges,
  noise-placement ends, imbalance ARI / F1_min, dominant failure mode).
- `run_all_v2.sh/.bat`: four final steps (tables, figures, claims audit).

## 2.0.0 — November 2026 (rebuild)

The whole code base was re-audited.
`framework-clustering-restructured/` (v1) is the ancestor of this repository;
every duplicate folder of the old archive (`framework/`, `TestRealData*`,
`Deeptest/`, `rabia/`) was either identical to it or older, and was dropped.

### Bugs fixed (these change results)

1. **The internal-noise fraction η_int was never applied in the noise sweep.**
   `exp2_sensitivity_analysis.py` tested `hasattr(config['noise'], 'internal_fraction')`;
   the field is `internal_noise_ratio`. All five η_int levels therefore ran on
   identical datasets (the v1 CSV shows a spread of exactly 0.0 in `D_dataset`
   and ARI across η_int). The "ARI is invariant to η_int" finding of the v1
   manuscript is an artefact of this bug. *Fixed; regression test
   `test_eta_int_changes_the_data`.*
2. **The `distant_noise` mask was always empty.** Noise points were split into
   internal/distant with `confidence <= noise_threshold = 0.05`, but a K-class
   posterior never falls below 1/K. Consequently `S_noise` depended on η only,
   and `D_dataset` was blind to the geometry of the noise. *Replaced by a
   geometric criterion (gap scale, `AmbiguityParams.distant_noise_quantile`);
   regression test `test_noise_split_masks_are_populated`.*
3. **"Internal" noise was not placed near cluster boundaries.** v1 drew it
   uniformly in the bounding box of the data enlarged by ×1.5; in d = 20 such
   points are far from every cluster, which is why density-based methods
   detected them trivially (ARI ≥ 0.999). *New default `internal_noise_mode =
   "bridge"` places each point on the segment between a cluster point and its
   nearest point in another cluster (+ scale-aware jitter). The legacy mode is
   kept as `"bounding_box"`.*
4. **The high-dimensional case study used a placeholder difficulty.**
   `exp5` set `D_dataset = 0.1·log(d+1)/log(501)` instead of computing it.
   *Now computes the real profile.*
5. **Evaluation convention.** v1 computed ARI/NMI on the points the
   *algorithm* kept (`labels != -1`), so a density method was scored only on
   what it chose to keep. `evaluate_partition` now reports `ARI` on the
   ground-truth inliers (rejecting a true inlier is an error), `ARI_full`
   (noise as a class), and `ARI_kept` (the v1 convention, for comparison),
   plus noise precision/recall and the number of clusters found.
6. **Θ_amb varied with the regime** (refinement disabled on *easy*,
   `core_threshold = 0.90` on *hard*), which made the three regimes
   incommensurable. The measurement instrument is now the same for every
   dataset: `profile_from_state` re-measures each STATE with the canonical
   `AmbiguityParams()`, and `difficulty_profile(X, y)` gives bit-identical
   results on the same data whatever its source.

### Design changes

- Confidence of labelled points is **out-of-sample** by default
  (`AmbiguityParams.confidence_mode = "cv"`, folds seeded from the data).
- **`clustaudit.metrics.profile.difficulty_profile`** is the single scoring
  chain. The stand-alone copy `ddataset_core.py` used by the real-data
  benchmark is gone (it had to be kept in sync by hand).
- **k-NN structural profile** (`clustaudit.metrics.knn_profile`): edge
  impurity, ambiguous fraction, fragmentation (mutual k-NN components),
  inter/intra density heterogeneity, noise proximity, TwoNN intrinsic
  dimension. Scale-free, no sphericity prior, O(n log n).
- **Baselines** are computed with every profile:
  1 − Silhouette(reference), cross-validated k-NN error, within/between ratio.
- **Tuning protocols** are explicit (`criterion = "silhouette" | "dbcv" | "fixed"`),
  selectable from the experiment scripts (`--criterion`).
- **Dimensionality control**: `--fixed-n` keeps n constant while d grows.
- **Audit interface**: `clustaudit.audit.audit(algo_fn, suite_dir)` and
  `sweep(algo_fn, factor, values)` run any `(X, seed) -> labels` callable on
  the suite or on a one-factor sweep (`anisotropy, eta, eta_int, overlap,
  connectivity, imbalance, inter_density, intra_density, dimension`).
- pip-installable package, CLI (`clustaudit generate | suite | profile`),
  pytest suite, no `sys.path` hacks inside the package.

### Kept as is

- The generator's geometry (hypercube centres, cluster shapes, densities,
  anisotropy, rotations) and the four v1 components of `D_dataset`.
- The v1 result CSVs under `benchmarks/*/results/` and `analysis/` scripts
  that rebuild the v1 tables from them (they document what was submitted;
  they will be regenerated by the new runs).

## 2.1.0 — analysis scripts

- `analysis/q1_ablation.py` (Table 3: LOO R² per predictor set, per family, bootstrap CIs),
  `analysis/q2_instrument_sensitivity.py` (Q2: Θ_amb perturbations + weight perturbations),
  `analysis/make_factors_figure.py` (Figure 2), `scripts/sweeps_v2.py` (overlap, imbalance,
  inter/intra density, connectivity sweeps), `scripts/run_all_v2.sh` (the whole pipeline in order).
- Pre-noise refinement never erases a class any more (heavy overlap could collapse the partition);
  `sweep()` skips a cell whose profile cannot be computed instead of aborting.
- 2.1.1: `O_rho` (diagnostic only) no longer divides by zero when a cluster has no core point
  (all its points are boundary, which happens in the hard regime). Values of the profile unchanged.

## 2.1.2 — density mechanisms, exp2 file naming
- **Density heterogeneity never ran** (v1 and v2.1): both `inter_density_variation` and the intra-cluster
  profiles were gated by `DensityParams.enable_adaptive_density`, default False, set nowhere. The gate is
  removed; the parameters now act whenever they are non-zero (regression test
  `test_density_parameters_change_the_data`).
- Intra-cluster density shaping is a monotone radial transform (v1 resampled points with replacement,
  which duplicated points). `sweep(factor="intra_density")` uses the `core_dense` mode.
- The fixed suite no longer sets density parameters (they had no effect; `datasets/suite_overlap`
  is bit-identical). Density is audited on its own sweep axes.
- `exp2 --only ...` writes one CSV per sub-sweep (`exp2_results_<criterion>_<subsweeps>.csv`), so
  successive runs no longer overwrite each other.
- `--criterion dbcv` now fails fast when `hdbscan` is missing instead of silently using the Silhouette.
- Hard regime of the suite: `cluster_std_range = (0.4, 0.6)` (real overlap).

## 2.1.3 — per-cluster recovery, failure modes, Silhouette-free scalar
- `evaluate_partition` adds `min_cluster_recall`, `min_cluster_f1`, `mean_cluster_f1` (recovery of the
  smallest reference cluster, which the pair-counting ARI hides under imbalance) and
  `failure_mode` in {ok, declared, silent}: a method that returns < 2 clusters or rejects most points
  *declares* its failure; one that returns a well-formed wrong partition fails *silently*.
- `DifficultyProfile.score_knn` (`D_knn` column): equal-weight mean of capped k-NN edge impurity,
  S_bnd and S_noise — no Silhouette, no shape prior, no fitted constant.
- (2.1.3 initially widened the rejection-sampling box for distant noise; this changed the outlier
  distribution in every dimension and was reverted in 2.1.4. The 2-D illustration uses a smaller
  `distant_noise_factor` instead.)

## 2.1.5 — DPC, cross-generator confrontation
- Density-peak clustering (`DensityPeaks`, `tune_dpc`, pool name "DPC") added to the pool; `exp1` and
  `scripts/sweeps_v2.py` accept `--algos` to run a subset (output gets a suffix).
- `benchmarks/cross-generator/run_crossgen.py` (MDCGen via mdcgenpy, HAWKS 0.2.0 with two compatibility
  shims) and `analyze_crossgen.py`; results committed under `benchmarks/cross-generator/results/`.
- `run_deep_clustering.py`: npz only (manifest.csv was loaded as a dataset), incremental save, `--resume`.

## 2.1.6 — control experiments
- Pre-noise refinement records `frac_relabelled` in the STATE metadata; `exp1 --no-prerefine` runs the suite
  on raw generated labels (output suffix `_noprerefine`); the fraction is a column of exp1 results.
- `scripts/sweeps_v2.py`: factors `eta`, `eta_int`, `anisotropy`; `--n` (absolute sizes), `--base-eta`,
  `--base-eta-int` (operating point overrides). Output suffixes record n and base eta.
- `analysis/audit_paper_claims_v2.py`: re-derives every number of the v2 manuscript from the result CSVs (92 claims); exits 1 on mismatch.
