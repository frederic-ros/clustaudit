# clustaudit — interventional auditing of clustering algorithms

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24239815.svg)](https://doi.org/10.5281/zenodo.24239815)

Code, pre-generated suite and result files for the paper

> F. Ros, R. Riad, Y. Nasser. *Interventional Auditing of Clustering Algorithms:
> Controlled Difficulty Sweeps Beyond Benchmarking.* (submitted)

A ranking says which method scored best on a dataset. An audit says under which
conditions a method fails, at what level, and in what form. `clustaudit` makes
that operational:

* a **generator** with seven structural controls (centre connectivity, overlap,
  anisotropy, inter- and intra-cluster density, size imbalance, dimension) and two
  separately adjustable noise controls (amount and placement);
* **STATE**, the structured ground truth returned with every dataset: reference
  membership, out-of-sample assignment confidence, core / boundary status and noise
  category of every point, measured after generation;
* a **difficulty profile** `H(X, y)`, computable from any labelled dataset with one
  fixed instrument, so that generated, independently generated and real data share
  coordinates;
* an **audit interface** `audit(method)` / `sweep(method, factor, values)` for any
  method `(X, seed) -> labels`, returning degradation curves, critical levels,
  failure modes (detectable / silent) and the fate of the smallest cluster.

Every result table and data figure of the paper and of its supplement is written by
a script of this repository from the result files shipped in `benchmarks/*/results/`,
and every number quoted in the text is re-derived from the same files by
`analysis/audit_paper_claims_v2.py` (243 checks).

---

## Installation

There are two ways to install, for two purposes.

**To use the package** (generate data, profile a dataset, audit a method): Python ≥ 3.9,
and the version ranges of `pyproject.toml`.

```bash
git clone https://github.com/frederic-ros/clustaudit.git
cd clustaudit
python -m pip install -e .
```

**To verify the paper's results**: the exact environment in which they were verified,
Python ≥ 3.12 with the pinned `requirements.txt` (or `environment.yml` for conda).

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The verification below also passes with the version ranges on Python 3.9, 3.10 and 3.11
(continuous integration runs both); the pinned environment is the reference.

Optional extras, only for the corresponding experiments:

| extra | packages | needed for |
|---|---|---|
| `dbcv`  | `hdbscan` | tuning protocol (D) |
| `deep`  | `clustpy`, `torch` | deep tier (IDEC, DCN, DipDECK) |
| `crossgen` | `mdcgenpy`, `hawks` | cross-generator confrontation |
| `figures` | `adjustText` | label placement in the real-data figure |

## Verifying the results: two modes

`scripts/reproduce.py` declares every step of the paper once, for every platform: its
command, the result files it writes and their expected number of rows (datasets ×
methods: no run may be missing). It has two modes, which certify different things.

**`verify-release`: check the distributed results (about 2 minutes).** No experiment is
re-run, and the report says so.

```bash
python scripts/reproduce.py verify-release        # or scripts/verify_release.sh / .bat
```

It runs the tests; checks that every shipped result file is complete and identical to
the SHA-256 recorded in `results_manifest.csv`; re-derives every number quoted in the
paper (`analysis/audit_paper_claims_v2.py`); and regenerates every result table, which
must be byte-identical to the paper's (`analysis/reference_tables/`), and the figures.

**`from-scratch`: re-run the experiments (hours).**

```bash
python scripts/reproduce.py from-scratch --out ../clustaudit_reproduction   # or scripts/run_all_v2.sh / .bat
```

It copies the code, without any result file and without the suite, into a new
directory and runs every experiment there. After each step it checks that the declared
outputs were written by that step with the expected number of rows; the first failure
stops the run. The analyses, the claims audit and the table comparison then run on the
new results. A step that cannot run (optional package missing, no network for the
real-data tier, or `--skip`) is reported as NOT RUN, and its outputs are copied from the
release and marked as such (`--strict`: not copied). `REPRODUCTION_REPORT.md`, in the
new directory, lists step by step and file by file what was reproduced and what was
taken from the release. Exit status: 0 full reproduction, 2 partial (some steps not
run, every check passed), 1 failure. `--resume` continues an interrupted run.

`python scripts/reproduce.py list` shows the steps and their outputs.

## Quickstart

```bash
python scripts/reproduce.py verify-release         # the whole check of the release (~2 min)
python analysis/audit_paper_claims_v2.py           # only the paper's numbers (~2 s)
python analysis/make_pipeline_figure.py            # Figure 3: one dataset through the pipeline (~5 s)
clustaudit profile --sklearn iris                  # difficulty profile of any labelled dataset
```

Audit your own method:

```python
from sklearn.cluster import KMeans
from clustaudit.audit import sweep, audit

my_method = lambda X, seed: KMeans(5, n_init=10, random_state=seed).fit_predict(X)

# one generator parameter varied, the others held fixed; descriptors of the data measured at every level
df = sweep(my_method, factor="eta_int", values=[0, .25, .5, .75, 1], seeds=[42, 123])   # ~3 s
print(df.groupby("value")[["ARI", "min_cluster_f1", "knn_noise_proximity"]].mean())

# the 75-dataset suite of the paper
df = audit(my_method, suite_dir="datasets/suite")                                        # ~15 s
```

Available sweep factors: `overlap`, `anisotropy`, `inter_density`, `intra_density`,
`imbalance`, `connectivity`, `dimension`, `eta`, `eta_int`.

## Repository layout

```
clustaudit/                 the package
  generators/               hypercube centres, clusters, noise, refinement, calibrate.py (Section 6.6)
  core/                     STATE and the generation pipeline (Algorithm 1)
  metrics/                  STATE components, k-NN descriptors, difficulty profile
  audit/                    algorithm pool, tuning protocols (S)/(D)/(F), evaluation, audit()/sweep()
benchmarks/
  classical/                suite audit (exp1), sensitivity grids (exp2), calibrated generation (exp6)
  deep/                     deep tier (IDEC, DCN, DipDECK)
  real-data/                19 UCI / OpenML datasets
  cross-generator/          MDCGen and HAWKS
  */results/                the result files behind every table and figure
scripts/                    reproduce.py (verify-release / from-scratch), sweeps_v2.py, wrappers .sh / .bat
analysis/                   paper tables, figures, claims audit, controls
datasets/suite/             the pre-generated 75-dataset suite (.npz) and its manifest
tests/                      generator and profile invariants, integration tests
results_manifest.csv        every shipped result file: producing step, rows, SHA-256
```

## Full reproduction

See `from-scratch` above. The classical experiments take about two hours on a laptop
(one thread per step), the deep tier about ten hours on CPU, and the real-data tier
downloads 15 datasets from OpenML. Seeds are fixed throughout (42, 123, 456, 789,
1011). The correspondence between each table of the paper and its script and result
files is given in Table S21 of the supplementary material and in `analysis/README.md`.

Every run is recorded. When no configuration of a tuning grid yields an admissible
partition (protocols S and D), the run is kept as a degenerate output, i.e. a
detectable failure (`run_status = no_admissible_configuration`); a method that raises
is kept as an error row (`failure_mode = error`, message in `run_error`).

The generator is deterministic across machines: the suite and every sweep regenerated
in the pinned environment are identical to machine precision to those of the original
runs (Windows / Anaconda, an earlier scikit-learn), and so are the difficulty profiles.
Method outputs are identical too, for every method except HDBSCAN and OPTICS, whose
scikit-learn implementations changed between versions. Over the 9,631 runs common to
both machines, they moved under protocol (F) on 51 runs, by at most 0.015 in ARI; under
(S) and (D), where a small change in a clustering can change the configuration the
criterion selects, on 154 runs, by up to 0.70 on a single run. The distributed results
are those of the pinned environment, and the paper quotes them; with another
scikit-learn version, expect these two methods to differ at that scale.

## Citation

Please cite the paper and the archived release you used (see also `CITATION.cff`):

* release v2.2.1 is the version described in the paper; the Zenodo record of all
  versions lists its DOI and resolves to the latest:
  [doi:10.5281/zenodo.24239815](https://doi.org/10.5281/zenodo.24239815);
* release v2.2.0, the first public release:
  [doi:10.5281/zenodo.24239816](https://doi.org/10.5281/zenodo.24239816).

## License

MIT, see `LICENSE`.
