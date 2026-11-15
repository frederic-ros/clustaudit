"""Integration tests: no run is ever lost, and the verification machinery fails when it should.

These tests exercise the paths that make the results certifiable: a tuning with no
admissible configuration, a method that raises, a sweep that contains both, the
claims audit on a corrupted result file, and the declared outputs of the
reproduction driver against the shipped results.
"""
import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from clustaudit import generate_dataset
from clustaudit.audit import (audit_dataset, evaluate_run, run_method, run_pool, sweep,
                              NoAdmissibleConfiguration, tuning_status, tune_dbscan)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _one_blob(n=200, d=20, seed=0):
    """Uniform points in a 20-dimensional cube (no cluster structure): every DBSCAN
    configuration of the Silhouette grid labels everything noise or one cluster, so no
    configuration is admissible."""
    rng = np.random.RandomState(seed)
    return rng.uniform(size=(n, d)), rng.randint(0, 2, n)


def _boom(X, seed):
    raise RuntimeError("deliberate failure")


def test_tuning_without_admissible_configuration_is_a_recorded_detectable_failure():
    X, y = _one_blob()
    model = tune_dbscan(X, None, None, "silhouette")
    assert isinstance(model, NoAdmissibleConfiguration) and model.n_tried == 20
    assert tuning_status(model) == "no_admissible_configuration"
    labels, status = run_method("DBSCAN", X, 0, "silhouette")
    row = evaluate_run(X, y, labels, status)
    assert row["run_status"] == "no_admissible_configuration"
    assert row["degenerate"] and row["failure_mode"] == "declared" and row["ARI"] == 0.0


def test_run_pool_never_drops_a_method():
    X, y = _one_blob()
    out = run_pool(X, 0, ["KMeans", "DBSCAN", "NoSuchMethod"], "silhouette", with_status=True)
    assert set(out) == {"KMeans", "DBSCAN", "NoSuchMethod"}
    assert out["DBSCAN"][1] == "no_admissible_configuration"
    assert out["NoSuchMethod"][0] is None and out["NoSuchMethod"][1].startswith("error")
    row = evaluate_run(X, y, *out["NoSuchMethod"])
    assert row["failure_mode"] == "error" and np.isnan(row["ARI"])


def test_audit_dataset_records_a_method_that_raises():
    st = generate_dataset(300, 5, 3, random_seed=1)
    rows = audit_dataset({"boom": _boom, "kmeans": lambda X, s: run_method("KMeans", X, s, "fixed", 3)},
                         st.Z, st.y, seed=1)
    assert [r["algorithm"] for r in rows] == ["boom", "kmeans"]
    assert rows[0]["failure_mode"] == "error" and "deliberate failure" in rows[0]["run_error"]
    assert rows[1]["run_status"] == "ok" and rows[1]["ARI"] > 0.9


def test_sweep_has_one_row_per_level_seed_and_method():
    algos = {"boom": _boom, "dbscan_S": lambda X, s: run_method("DBSCAN", X, s, "silhouette"),
             "kmeans_F": lambda X, s: run_method("KMeans", X, s, "fixed", 5)}
    df = sweep(algos, factor="eta_int", values=[0.0, 1.0], seeds=[42], n_samples=400, dimension=6)
    assert len(df) == 2 * 1 * 3
    assert (df[df.algorithm == "boom"].failure_mode == "error").all()
    assert set(df.run_status) <= {"ok", "no_admissible_configuration", "error"}


def _load_driver():
    spec = importlib.util.spec_from_file_location("reproduce", os.path.join(ROOT, "scripts", "reproduce.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reproduce"] = mod            # needed by @dataclass
    spec.loader.exec_module(mod)
    return mod


def test_shipped_result_files_match_the_declared_steps():
    """Every output declared by the reproduction driver is shipped, complete (no run
    missing) and identical to the SHA-256 recorded in results_manifest.csv."""
    drv = _load_driver()
    manifest = drv.read_manifest(ROOT)
    if manifest is None:
        pytest.skip("no results_manifest.csv (development checkout)")
    problems = []
    for s in drv.STEPS:
        for rel, expected in s.outputs:
            ok, msg = drv.check_file(ROOT, rel, expected, manifest=manifest)
            if not ok:
                problems.append(f"{rel}: {msg}")
    assert not problems, problems


def test_driver_rejects_a_file_with_a_missing_run(tmp_path):
    drv = _load_driver()
    rel = "benchmarks/classical/results/sweeps_v2/sweep_eta_int_fixed_eta0.3.csv"
    src = os.path.join(ROOT, rel)
    if not os.path.exists(src):
        pytest.skip("result files not present")
    dst = tmp_path / rel
    dst.parent.mkdir(parents=True)
    pd.read_csv(src).iloc[:-1].to_csv(dst, index=False)       # one run removed
    ok, msg = drv.check_file(str(tmp_path), rel, 225)
    assert not ok and "224 rows" in msg


def test_claims_audit_fails_on_an_altered_result(tmp_path):
    """The claims audit must exit with status 1 when a result file no longer supports a
    number of the paper."""
    src = os.path.join(ROOT, "benchmarks")
    if not os.path.exists(os.path.join(src, "classical", "results", "sweeps_v2", "sweep_eta_int_fixed_eta0.3.csv")):
        pytest.skip("result files not present")
    shutil.copytree(src, tmp_path / "benchmarks", ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    if os.path.isdir(os.path.join(ROOT, "analysis", "tables")):
        shutil.copytree(os.path.join(ROOT, "analysis", "tables"), tmp_path / "analysis" / "tables")
    f = tmp_path / "benchmarks/classical/results/sweeps_v2/sweep_eta_int_fixed_eta0.3.csv"
    d = pd.read_csv(f)
    d.loc[d.algorithm == "KMeans", "ARI"] = 0.0                  # the noise crossover disappears for K-Means
    d.to_csv(f, index=False)
    p = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "audit_paper_claims_v2.py"),
                        "--results-root", str(tmp_path)], capture_output=True, text=True)
    assert p.returncode == 1 and "MISMATCH" in p.stdout
