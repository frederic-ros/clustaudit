#!/usr/bin/env python
"""
Verification and reproduction driver of clustaudit: one list of steps, every platform.

    python scripts/reproduce.py verify-release              # check the distributed results
    python scripts/reproduce.py from-scratch --out DIR      # re-run the experiments in DIR, then check
    python scripts/reproduce.py list                        # steps, outputs, expected sizes

verify-release (minutes; no experiment is re-run, and the report says so)
  1. runs the test suite;
  2. checks every result file declared below: present, expected number of rows, no run
     missing from it, and the SHA-256 recorded in results_manifest.csv (unaltered);
  3. re-derives every number quoted in the paper (analysis/audit_paper_claims_v2.py);
  4. regenerates every result table and compares it byte for byte with the tables of the
     paper (analysis/reference_tables/), and regenerates the figures.

from-scratch (hours)
  copies the code, without any result file and without the suite, into the new
  directory DIR and runs every experiment there. After each step it checks that each
  declared output was written during that step and has the expected number of rows; the
  first failure of a step stops the run (--keep-going records it and continues). A step
  that cannot run (optional package missing, or excluded with --skip) is NOT RUN: its
  outputs are copied from the release and marked as taken from it; with --strict nothing
  is copied and the analyses that need those files fail. The analyses, the claims audit
  and the table comparison then run on DIR. DIR/REPRODUCTION_REPORT.md says, step by step
  and file by file, what was reproduced and what was taken from the release.

Exit status: 0 = every check passed (from-scratch: and every step was reproduced);
             2 = from-scratch checks passed, but some steps were not run (partial);
             1 = a check or a step failed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
PY = sys.executable
E1 = "benchmarks/classical/results/exp1_difficulty_validation"
E2 = "benchmarks/classical/results/exp2_sensitivity_analysis"
E6 = "benchmarks/classical/results/exp6_calibrated_generation"
SW = "benchmarks/classical/results/sweeps_v2"
MANIFEST = "results_manifest.csv"
REFERENCE_TABLES = "analysis/reference_tables"


@dataclass
class Step:
    name: str
    cmd: List[str]
    outputs: List[Tuple[str, Optional[int]]] = field(default_factory=list)   # (path, expected rows)
    optional: Optional[str] = None          # module whose absence makes the step optional-skipped
    kind: str = "experiment"                # "experiment" | "analysis"
    note: str = ""


def _py(*args):
    return [PY] + list(args)


SWEEP = "scripts/sweeps_v2.py"
EXP1 = "benchmarks/classical/exp1_difficulty_validation.py"
EXP2 = "benchmarks/classical/exp2_sensitivity_analysis.py"

# Every result file the analyses read is the declared output of exactly one step.
# Expected rows = datasets x methods: no run may be missing (a tuning without an admissible
# configuration and a method that raises are recorded rows, see clustaudit.audit).
STEPS: List[Step] = [
    Step("suite", _py("-m", "clustaudit.cli", "suite", "--out", "datasets/suite"),
         [("datasets/suite/manifest.csv", 75)], note="the 75-dataset suite"),
    # suite audit under the three tuning protocols (10 methods x 75 datasets)
    Step("exp1_S", _py(EXP1, "--criterion", "silhouette"), [(f"{E1}/exp1_results.csv", 750)]),
    Step("exp1_F", _py(EXP1, "--criterion", "fixed"), [(f"{E1}/exp1_results_fixed.csv", 750)]),
    Step("exp1_D", _py(EXP1, "--criterion", "dbcv"), [(f"{E1}/exp1_results_dbcv.csv", 750)], optional="hdbscan"),
    Step("exp1_F_noprerefine", _py(EXP1, "--criterion", "fixed", "--no-prerefine"),
         [(f"{E1}/exp1_results_fixed_noprerefine.csv", 750)]),
    Step("ctrl_refine_svm", _py(EXP1, "--criterion", "fixed", "--regime", "hard", "--refine-classifier", "svm"),
         [(f"{E1}/exp1_results_fixed_refinesvm_hard.csv", 250)]),
    Step("ctrl_refine_rf", _py(EXP1, "--criterion", "fixed", "--regime", "hard", "--refine-classifier", "random_forest"),
         [(f"{E1}/exp1_results_fixed_refinerandom_forest_hard.csv", 250)]),
    # sensitivity grids (8 classical methods)
    Step("exp2_noise_S", _py(EXP2, "--only", "noise", "--out", "exp2_results_silhouette_noise.csv"),
         [(f"{E2}/exp2_results_silhouette_noise.csv", 1800)]),
    Step("exp2_aniso_dim_S", _py(EXP2, "--only", "anisotropy", "dimensionality", "--out", "exp2_results.csv"),
         [(f"{E2}/exp2_results.csv", 720)]),
    Step("exp2_dim_fixedn_S", _py(EXP2, "--only", "dimensionality", "--fixed-n", "--out", "exp2_results_silhouette_fixedn.csv"),
         [(f"{E2}/exp2_results_silhouette_fixedn.csv", 360)]),
    Step("exp2_aniso_F", _py(EXP2, "--criterion", "fixed", "--only", "anisotropy", "--out", "exp2_results_fixed_anisotropy.csv"),
         [(f"{E2}/exp2_results_fixed_anisotropy.csv", 360)]),
    # single-factor sweeps (9 methods, five seeds)
    Step("sweeps_S", _py(SWEEP, "--factor", "overlap", "imbalance", "inter_density", "intra_density", "connectivity",
                         "--criterion", "silhouette"),
         [(f"{SW}/sweep_overlap_silhouette.csv", 315), (f"{SW}/sweep_imbalance_silhouette.csv", 225),
          (f"{SW}/sweep_inter_density_silhouette.csv", 225), (f"{SW}/sweep_intra_density_silhouette.csv", 180),
          (f"{SW}/sweep_connectivity_silhouette.csv", 225)]),
    Step("sweeps_F", _py(SWEEP, "--factor", "overlap", "imbalance", "inter_density", "intra_density", "--criterion", "fixed"),
         [(f"{SW}/sweep_overlap_fixed.csv", 315), (f"{SW}/sweep_imbalance_fixed.csv", 225),
          (f"{SW}/sweep_inter_density_fixed.csv", 225), (f"{SW}/sweep_intra_density_fixed.csv", 180)]),
    Step("sweep_connectivity_F_DPC", _py(SWEEP, "--factor", "connectivity", "--criterion", "fixed", "--algos", "DPC"),
         [(f"{SW}/sweep_connectivity_fixed_DPC.csv", 25)],
         note="DPC is not in the anisotropy grid; its operating envelope uses this sweep instead"),
    Step("sweep_imb_n3000", _py(SWEEP, "--factor", "imbalance", "--criterion", "fixed", "--n", "3000"),
         [(f"{SW}/sweep_imbalance_fixed_n3000.csv", 225)]),
    Step("sweep_eta_int_F", _py(SWEEP, "--factor", "eta_int", "--base-eta", "0.3", "--criterion", "fixed"),
         [(f"{SW}/sweep_eta_int_fixed_eta0.3.csv", 225)]),
    # controls
    Step("ctrl_raw_eta", _py(SWEEP, "--factor", "eta_int", "--base-eta", "0.3", "--criterion", "fixed", "--no-prerefine"),
         [(f"{SW}/sweep_eta_int_fixed_eta0.3_noprerefine.csv", 225)]),
    Step("ctrl_raw_imb", _py(SWEEP, "--factor", "imbalance", "--criterion", "fixed", "--no-prerefine"),
         [(f"{SW}/sweep_imbalance_fixed_noprerefine.csv", 225)]),
] + [
    Step(f"ctrl_K{K}_{f}", _py(SWEEP, "--factor", f if f == "imbalance" else "eta_int", *(["--base-eta", "0.3"] if f == "eta" else []),
                               "--criterion", "fixed", "--K", str(K)),
         [(f"{SW}/sweep_{'imbalance_fixed' if f == 'imbalance' else 'eta_int_fixed_eta0.3'}_K{K}.csv", 225)])
    for K in (3, 8) for f in ("eta", "imbalance")
] + [
    Step("ctrl_spectral", _py("analysis/controls_v2.py", "--check-spectral"), [(f"{SW}/spectral_imbalance_check.csv", 5)]),
    Step("exp6_calibrated", _py("benchmarks/classical/exp6_calibrated_generation.py"),
         [(f"{E6}/datasets.csv", 80), (f"{E6}/pool.csv", 720)]),
    # the three tiers that need more than the core install
    Step("crossgen", _py("benchmarks/cross-generator/run_crossgen.py"),
         [("benchmarks/cross-generator/results/crossgen_results.csv", 1260)], optional="hawks",
         note="needs hawks and mdcgenpy (pip install -e .[crossgen])"),
    Step("deep", _py("benchmarks/deep/run_deep_clustering.py", "--data_dir", "datasets/suite"),
         [("benchmarks/deep/results/deep_results.csv", 1125)], optional="clustpy",
         note="needs clustpy and torch (pip install -e .[deep]); about ten hours on CPU"),
    Step("real_raw", _py("benchmarks/real-data/run_transfer.py"),
         [("benchmarks/real-data/results/transfer_results.csv", 19), ("benchmarks/real-data/results/transfer_per_algo.csv", 152)],
         note="downloads 15 datasets from OpenML (network access)"),
    Step("real_relabel", _py("benchmarks/real-data/run_transfer.py", "--relabel"),
         [("benchmarks/real-data/results/transfer_results_relabel.csv", 19)], note="network access, as real_raw"),
    Step("q2_instrument", _py("analysis/q2_instrument_sensitivity.py", "--suite", "datasets/suite"),
         [("analysis/tables/q2_instrument_sensitivity.csv", 11)]),
    # analyses: they read the files above and write tables, figures and two summary files
    Step("controls", _py("analysis/controls_v2.py"), kind="analysis"),
    Step("audit_value", _py("analysis/audit_value_v2.py"),
         [("benchmarks/cross-generator/results/predictive_summary.csv", 24), (f"{SW}/paired_ratios.csv", 40)], kind="analysis"),
    Step("paper_tables", _py("analysis/make_paper_tables_v2.py"), kind="analysis"),
    Step("paper_figures", _py("analysis/make_paper_figures_v2.py"), kind="analysis"),
    Step("pipeline_figure", _py("analysis/make_pipeline_figure.py"), kind="analysis"),
]
STEP_BY_NAME = {s.name: s for s in STEPS}


# ----------------------------------------------------------------------------- helpers
TEXT_RESULTS = (".csv", ".tex", ".md", ".txt")


def sha256(path):
    """SHA-256 of a file; for text files, of its content with line endings normalized to LF,
    so that a checkout converting LF to CRLF (git on Windows) does not count as an alteration."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        data = f.read()
    if path.lower().endswith(TEXT_RESULTS):
        data = data.replace(b"\r\n", b"\n")
    h.update(data)
    return h.hexdigest()


def n_rows(path):
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)        # data rows of a CSV (header excluded)


def have(module):
    return importlib.util.find_spec(module) is not None


def run(cmd, cwd, log, env=None, timeout=None):
    # The scripts print non-ASCII text (arrows, Greek letters). Their output goes to a log file,
    # which Python encodes in the locale's code page by default (cp1252 on Windows, where they
    # would then fail): force UTF-8 on every platform.
    env = dict(os.environ if env is None else env, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return p.returncode, time.time() - t0


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def env_report():
    lines = [f"- Python {platform.python_version()} ({platform.platform()})"]
    for m in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "hdbscan", "clustpy", "torch", "hawks", "mdcgenpy"):
        try:
            mod = __import__(m)
            lines.append(f"- {m} {getattr(mod, '__version__', '?')}")
        except Exception:
            lines.append(f"- {m}: not installed")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if commit:
            lines.append(f"- clustaudit commit {commit}")
    except Exception:
        pass
    return lines


def check_file(base, rel, expected, t_start=None, manifest=None):
    """Return (ok, message) for one declared output."""
    p = os.path.join(base, rel)
    if not os.path.exists(p):
        return False, "missing"
    if t_start is not None and os.path.getmtime(p) < t_start - 1:
        return False, "not written by this step (older file)"
    r = n_rows(p)
    if expected is not None and r != expected:
        return False, f"{r} rows, expected {expected}"
    if manifest is not None:
        h = manifest.get(rel)
        if h is None:
            return False, "not in results_manifest.csv"
        if sha256(p) != h:
            return False, "SHA-256 differs from results_manifest.csv"
    return True, f"{r} rows"


def _same_numbers(p, q, tol=1e-9):
    """Two CSV files with the same layout and the same numbers up to tol (text cells equal)."""
    import pandas as pd
    a, b = pd.read_csv(p, float_precision="round_trip"), pd.read_csv(q, float_precision="round_trip")
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if not ((a[c] - b[c]).abs().fillna(0) <= tol).all() or not (a[c].isna() == b[c].isna()).all():
                return False
        elif not (a[c].astype(str) == b[c].astype(str)).all():
            return False
    return True


def compare_tables(new_dir, ref_dir):
    ref = sorted(f for f in os.listdir(ref_dir) if f.endswith(".tex")) if os.path.isdir(ref_dir) else []
    same, diff, missing = [], [], []
    for f in ref:
        p = os.path.join(new_dir, f)
        if not os.path.exists(p):
            missing.append(f)
        elif open(p, "rb").read().replace(b"\r\n", b"\n") == open(os.path.join(ref_dir, f), "rb").read().replace(b"\r\n", b"\n"):
            same.append(f)
        else:
            diff.append(f)
    return ref, same, diff, missing


def read_manifest(base):
    p = os.path.join(base, MANIFEST)
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 4:
                out[parts[0]] = parts[3]
    return out


def write_manifest(base):
    """results_manifest.csv: path, producing step, rows, SHA-256 of every declared output."""
    rows = []
    for s in STEPS:
        for rel, _ in s.outputs:
            p = os.path.join(base, rel)
            if os.path.exists(p):
                rows.append(f"{rel},{s.name},{n_rows(p)},{sha256(p)}")
            else:
                print(f"  ! {rel} missing, not in the manifest")
    suite = os.path.join(base, "datasets", "suite")
    for f in sorted(os.listdir(suite)) if os.path.isdir(suite) else []:
        if f.endswith(".npz"):
            rows.append(f"datasets/suite/{f},suite,,{sha256(os.path.join(suite, f))}")
    with open(os.path.join(base, MANIFEST), "w", encoding="utf-8") as fh:
        fh.write("path,step,rows,sha256\n" + "\n".join(rows) + "\n")
    print(f"{MANIFEST}: {len(rows)} files")


# ----------------------------------------------------------------------------- verify-release
def verify_release(a):
    base = os.path.abspath(a.results_root)
    report, ok_all = ["# clustaudit: verification of the distributed results", "",
                      f"{_dt.datetime.now():%Y-%m-%d %H:%M}. Mode: **verify-release**. No experiment was re-run: this checks the "
                      "result files shipped with the release, the numbers the paper derives from them, and the tables "
                      "and figures regenerated from them.", "", "## Environment", *env_report(), ""], True
    logs = tempfile.mkdtemp(prefix="clustaudit_verify_")

    # 1. tests
    if not a.no_tests:
        rc, dt = run(_py("-m", "pytest", "-q", "tests"), base, os.path.join(logs, "pytest.log"))
        tail = _read(os.path.join(logs, "pytest.log")).strip().splitlines()[-1:] or [""]
        report += ["## 1. Test suite", f"- {'PASS' if rc == 0 else 'FAIL'}: {tail[0]} ({dt:.0f} s)", ""]
        ok_all &= rc == 0

    # 2. result files
    manifest = read_manifest(base)
    report += ["## 2. Result files", "| file | step | check |", "|---|---|---|"]
    if manifest is None:
        report.append(f"| {MANIFEST} | | FAIL: missing |"); ok_all = False
    n_ok = 0
    for s in STEPS:
        for rel, exp in s.outputs:
            good, msg = check_file(base, rel, exp, manifest=manifest)
            n_ok += good; ok_all &= good
            report.append(f"| `{rel}` | {s.name} | {'ok' if good else 'FAIL'}: {msg} |")
    declared = {rel for s in STEPS for rel, _ in s.outputs}
    others = {k: v for k, v in (manifest or {}).items() if k not in declared}
    bad = [k for k, h in others.items() if not os.path.exists(os.path.join(base, k)) or sha256(os.path.join(base, k)) != h]
    report.append(f"| suite (`datasets/suite/*.npz`) | suite | {'ok' if not bad else 'FAIL'}: "
                  f"{len(others) - len(bad)}/{len(others)} files identical to the manifest |")
    ok_all &= not bad and len(others) > 0
    report.append("")

    # 3. claims
    rc, dt = run(_py("analysis/audit_paper_claims_v2.py", "--results-root", base), base, os.path.join(logs, "claims.log"))
    tail = [l for l in _read(os.path.join(logs, "claims.log")).splitlines() if "claims match" in l]
    bad = [l for l in _read(os.path.join(logs, "claims.log")).splitlines() if "MISMATCH" in l or "NOT COMPUTED" in l]
    report += ["## 3. Numbers quoted in the paper", f"- {'PASS' if rc == 0 else 'FAIL'}: {tail[0] if tail else 'no summary line'}"]
    report += [f"  - {l.strip()}" for l in bad] + [""]
    ok_all &= rc == 0

    # 4. tables and figures, from a temporary copy of the result files (the analyses rewrite two
    #    summary files; the shipped tree is never written to)
    tdir, fdir, work = os.path.join(logs, "tables"), os.path.join(logs, "figures"), os.path.join(logs, "results_copy")
    for d in ("benchmarks", "analysis/tables"):
        if os.path.isdir(os.path.join(base, d)):
            shutil.copytree(os.path.join(base, d), os.path.join(work, d),
                            ignore=lambda _d, names: [n for n in names if n.endswith(".py") or n == "__pycache__"])
    steps = [("controls", _py("analysis/controls_v2.py", "--results-root", work, "--out", tdir)),
             ("audit_value", _py("analysis/audit_value_v2.py", "--results-root", work, "--out", tdir)),
             ("paper_tables", _py("analysis/make_paper_tables_v2.py", "--results-root", work, "--out", tdir))]
    if not a.no_figures:
        steps += [("paper_figures", _py("analysis/make_paper_figures_v2.py", "--results-root", work, "--out", fdir)),
                  ("pipeline_figure", _py("analysis/make_pipeline_figure.py", "--out", fdir))]
    report += ["## 4. Tables and figures"]
    for name, cmd in steps:
        rc, dt = run(cmd, base, os.path.join(logs, f"{name}.log"))
        report.append(f"- {name}: {'ok' if rc == 0 else 'FAIL'} ({dt:.0f} s)"); ok_all &= rc == 0
    ref, same, diff, missing = compare_tables(tdir, os.path.join(base, REFERENCE_TABLES))
    report.append(f"- tables identical to the paper's: {len(same)}/{len(ref)}"
                  + (f"; different: {', '.join(diff)}" if diff else "") + (f"; not produced: {', '.join(missing)}" if missing else ""))
    ok_all &= bool(ref) and not diff and not missing
    if not a.no_figures:
        figs = sorted(f for f in os.listdir(fdir) if f.endswith(".pdf")) if os.path.isdir(fdir) else []
        report.append(f"- figures regenerated: {len(figs)} ({', '.join(figs)})")
    # the two summary files regenerated by audit_value must equal the shipped ones
    if manifest:
        for rel in ("benchmarks/cross-generator/results/predictive_summary.csv", f"{SW}/paired_ratios.csv"):
            p, q = os.path.join(work, rel), os.path.join(base, rel)
            same_ = os.path.exists(p) and _same_numbers(p, q)
            report.append(f"- regenerated `{os.path.basename(rel)}` equal to the shipped one (to 1e-9): {'yes' if same_ else 'NO (FAIL)'}")
            ok_all &= same_
    # 5. release metadata: no placeholder may remain in what a user or Zenodo reads
    report += ["", "## 5. Release metadata"]
    left = []
    for f in ("README.md", "pyproject.toml", "CITATION.cff", ".zenodo.json"):
        p = os.path.join(base, f)
        if os.path.exists(p):
            txt = open(p, encoding="utf-8").read()
            left += [f"`{f}`: {tok}" for tok in ("<owner>", "NNNNNNN", "TODO") if tok in txt]
    report.append("- no placeholder left" if not left else "- WARNING, placeholders left (fill them before the release): " + "; ".join(left))
    report += ["", "## Verdict",
               ("**PASS.** The distributed results are complete and unaltered, every number quoted in the paper is "
                "re-derived from them, and every table of the paper is regenerated identically. This does not re-run "
                "the experiments: `python scripts/reproduce.py from-scratch --out DIR` does.") if ok_all else
               "**FAIL.** See the items marked FAIL above; logs in " + logs, ""]
    out = a.report or os.path.join(logs, "VERIFICATION_REPORT.md")
    open(out, "w", encoding="utf-8").write("\n".join(report))
    print("\n".join(report)); print(f"report: {out}")
    return 0 if ok_all else 1


# ----------------------------------------------------------------------------- from-scratch
COPY_EXCLUDE = {".git", "logs", "__pycache__", ".pytest_cache", "reproduction", "tables_paper", "figures_paper"}


def _copy_code(dst):
    def ignore(d, names):
        rel = os.path.relpath(d, ROOT)
        out = [n for n in names if n in COPY_EXCLUDE or n.endswith((".egg-info", ".pyc"))]
        if os.path.abspath(d) == os.path.abspath(os.path.dirname(dst)):
            out.append(os.path.basename(dst))
        if os.path.basename(d) == "results" or rel.replace("\\", "/") in ("datasets", "analysis/tables", "analysis/figures"):
            out += [n for n in names if n not in out and not n.startswith(".git")]   # no result file, no suite
        return out
    shutil.copytree(ROOT, dst, ignore=ignore, dirs_exist_ok=True)


def from_scratch(a):
    out = os.path.abspath(a.out)
    state_p = os.path.join(out, ".reproduce_state.json")
    if os.path.exists(out) and os.listdir(out) and not a.resume:
        sys.exit(f"{out} exists and is not empty (use a new directory, or --resume to continue a previous run)")
    if os.path.commonpath([out, ROOT]) == ROOT and not os.path.relpath(out, ROOT).startswith("reproduction"):
        sys.exit("DIR must be outside the repository, or under ./reproduction")
    if not a.resume or not os.path.exists(state_p):
        _copy_code(out)
        json.dump({"done": {}}, open(state_p, "w"))
    state = json.load(open(state_p))
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    env = dict(os.environ, PYTHONPATH=out + os.pathsep + os.environ.get("PYTHONPATH", ""), MPLBACKEND="Agg")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # see --threads
        env[var] = str(a.threads)

    sel = lambda s, pats: any(fnmatch.fnmatch(s.name, p) for p in pats)
    status, failed = {}, False
    for s in STEPS:
        if s.kind == "experiment" and a.only and not sel(s, a.only):     # analyses always run
            reason = "not selected (--only)"
        elif s.kind == "experiment" and a.skip and sel(s, a.skip):
            reason = "excluded (--skip)"
        elif s.optional and not have(s.optional):
            reason = f"optional package `{s.optional}` not installed"
        elif failed and not a.keep_going:
            reason = "not run: an earlier step failed"
        else:
            reason = None
        if reason is None and s.name in state["done"] and a.resume:
            status[s.name] = ("REPRODUCED", "in an earlier session of this run (--resume)"); continue
        if reason is not None:
            if s.kind == "experiment" and not a.strict and "earlier step failed" not in reason:
                copied = []
                if s.name == "suite" and os.path.isdir(os.path.join(ROOT, "datasets", "suite")):
                    shutil.copytree(os.path.join(ROOT, "datasets", "suite"), os.path.join(out, "datasets", "suite"), dirs_exist_ok=True)
                for rel, _ in s.outputs:
                    src = os.path.join(ROOT, rel)
                    if os.path.exists(src):
                        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
                        shutil.copy2(src, os.path.join(out, rel)); copied.append(rel)
                status[s.name] = ("NOT RUN", reason + ("; outputs taken from the release" if copied else ""))
            else:
                status[s.name] = ("NOT RUN", reason)
            print(f"[{_dt.datetime.now():%H:%M:%S}] {s.name:24s} NOT RUN ({status[s.name][1]})", flush=True)
            continue
        print(f"[{_dt.datetime.now():%H:%M:%S}] {s.name:24s} ...", flush=True)
        t0 = time.time()
        rc, dt = run(s.cmd, out, os.path.join(out, "logs", f"{s.name}.log"), env=env)
        problems = [f"exit status {rc}"] if rc != 0 else []
        for rel, exp in s.outputs:
            good, msg = check_file(out, rel, exp, t_start=t0)
            if not good:
                problems.append(f"{rel}: {msg}")
        if problems:
            status[s.name] = ("FAILED", "; ".join(problems) + f" (log: logs/{s.name}.log)"); failed = True
        else:
            status[s.name] = ("REPRODUCED", f"{dt / 60:.1f} min")
            state["done"][s.name] = {rel: sha256(os.path.join(out, rel)) for rel, _ in s.outputs}
            json.dump(state, open(state_p, "w"), indent=1)
        print(f"[{_dt.datetime.now():%H:%M:%S}] {s.name:24s} {status[s.name][0]} ({status[s.name][1]})", flush=True)

    # claims and table comparison on DIR
    claims_ok, claims_line, claims_bad = False, "not run", []
    if not failed or a.keep_going:
        rc, _ = run(_py("analysis/audit_paper_claims_v2.py"), out, os.path.join(out, "logs", "claims.log"), env=env)
        txt = _read(os.path.join(out, "logs", "claims.log")).splitlines()
        claims_ok = rc == 0
        claims_line = next((l for l in txt if "claims match" in l), "no summary line")
        claims_bad = [l.strip() for l in txt if "MISMATCH" in l or "NOT COMPUTED" in l]
    ref, same, diff, missing = compare_tables(os.path.join(out, "analysis", "tables_paper"), os.path.join(ROOT, REFERENCE_TABLES))

    n_rep = sum(v[0] == "REPRODUCED" for v in status.values())
    not_run = [k for k, v in status.items() if v[0] == "NOT RUN"]
    n_fail = sum(v[0] == "FAILED" for v in status.values())
    full = n_fail == 0 and not not_run and claims_ok
    verdict = ("**FULL REPRODUCTION.** Every step was re-run in this directory and every number quoted in the paper "
               "is re-derived from the new results." if full else
               "**PARTIAL REPRODUCTION.** The steps marked NOT RUN were not re-run; where their outputs were taken "
               "from the release, the numbers that depend on them are checked against the distributed files, not "
               "reproduced. Every other step was re-run and the checks passed." if (n_fail == 0 and claims_ok) else
               "**FAILED.** See the steps marked FAILED and the claims audit.")
    rep = ["# clustaudit: reproduction from scratch", "",
           f"{_dt.datetime.now():%Y-%m-%d %H:%M}. Mode: **from-scratch**, in `{out}` (code copied without any "
           "result file and without the suite).", "", "## Environment", *env_report(), "",
           f"## Steps ({n_rep} reproduced, {len(not_run)} not run, {n_fail} failed)", "| step | status | detail |", "|---|---|---|"]
    rep += [f"| {k} | {v[0]} | {v[1]} |" for k, v in status.items()]
    rep += ["", "## Result files", "| file | origin | rows | SHA-256 (first 12) | same as release |", "|---|---|---|---|---|"]
    man = read_manifest(ROOT) or {}
    for s in STEPS:
        for rel, _ in s.outputs:
            p = os.path.join(out, rel)
            if not os.path.exists(p):
                rep.append(f"| `{rel}` | absent | | | |"); continue
            origin = "reproduced" if status.get(s.name, ("",))[0] == "REPRODUCED" else "taken from the release"
            h = sha256(p)
            rep.append(f"| `{rel}` | {origin} | {n_rows(p)} | {h[:12]} | {'yes' if man.get(rel) == h else 'no'} |")
    rep += ["", "## Numbers quoted in the paper", f"- {'PASS' if claims_ok else 'FAIL'}: {claims_line}"] + [f"  - {l}" for l in claims_bad]
    rep += ["", "## Tables", f"- identical to the paper's: {len(same)}/{len(ref)}"
            + (f"; different: {', '.join(diff)} (see `diff analysis/tables_paper/<file> {REFERENCE_TABLES}/<file>`)" if diff else "")
            + (f"; not produced: {', '.join(missing)}" if missing else ""),
            "- a result file that is not byte-identical to the release (column 'same as release') is expected where an "
            "implementation differs between library versions (HDBSCAN, see README); the claims audit decides whether "
            "any number of the paper is affected.", "", "## Verdict", verdict, ""]
    open(os.path.join(out, "REPRODUCTION_REPORT.md"), "w", encoding="utf-8").write("\n".join(rep))
    print("\n".join(rep))
    return 0 if full else (2 if (n_fail == 0 and claims_ok) else 1)


# ----------------------------------------------------------------------------- main
def main():
    if hasattr(sys.stdout, "reconfigure"):          # a console or file that cannot show a character
        sys.stdout.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("verify-release", help="check the distributed results (no experiment is re-run)")
    v.add_argument("--results-root", default=ROOT)
    v.add_argument("--report", default=None, help="where to write the Markdown report")
    v.add_argument("--no-tests", action="store_true"); v.add_argument("--no-figures", action="store_true")
    f = sub.add_parser("from-scratch", help="re-run the experiments in a new directory, then check")
    f.add_argument("--out", required=True, help="new directory (outside the repository, or under ./reproduction)")
    f.add_argument("--only", nargs="*", default=None, help="run only these experiment steps (glob patterns); the other experiments are NOT RUN, the analyses always run")
    f.add_argument("--skip", nargs="*", default=None, help="do not run these experiment steps (glob patterns)")
    f.add_argument("--strict", action="store_true", help="never copy the outputs of a step that is not run")
    f.add_argument("--keep-going", action="store_true", help="continue after a failed step (the run still fails)")
    f.add_argument("--resume", action="store_true", help="continue a previous run in the same directory")
    f.add_argument("--threads", type=int, default=1,
                   help="BLAS/OpenMP threads per step (default 1: avoids oversubscription when several runs share a machine)")
    sub.add_parser("list", help="list the steps, their outputs and expected sizes")
    m = sub.add_parser("manifest", help="(maintainers) write results_manifest.csv from the current result files")
    m.add_argument("--results-root", default=ROOT)
    a = ap.parse_args()
    if a.mode == "list":
        for s in STEPS:
            opt = f"  [optional: {s.optional}]" if s.optional else ""
            print(f"{s.name:26s} {s.kind:10s}{opt}  {' '.join(os.path.relpath(c, ROOT) if os.path.isabs(c) and c != PY else c for c in s.cmd[1:])}")
            for rel, exp in s.outputs:
                print(f"{'':26s}   -> {rel} ({exp} rows)")
        return 0
    if a.mode == "manifest":
        write_manifest(os.path.abspath(a.results_root)); return 0
    return verify_release(a) if a.mode == "verify-release" else from_scratch(a)


if __name__ == "__main__":
    sys.exit(main())
