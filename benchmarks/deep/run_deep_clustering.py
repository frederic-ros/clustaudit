#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep tier of the audit: IDEC, DCN (K given) and DipDECK (K estimated), via ClustPy.

For every dataset of a folder (or a small synthetic demo set) the script runs the
three methods with their reference defaults, five runs each, and evaluates every
partition with the same function as the classical tier
(clustaudit.audit.evaluation.evaluate_partition): ARI and NMI on the ground-truth
inliers (the reference convention of the paper), ARI and NMI with noise as a class
(ARI_full, the convention of versions <= 2.2.1 of this script), on the kept points,
the internal indices, the number of clusters found, the smallest-cluster F1 and the
failure mode. Each method is judged on its label vector only (black box); internal
indices are computed in the input space X, never in the learned embedding, so that
they are comparable with the classical tier. Every label vector is saved, so that
another metric can be computed later without re-running the methods, and the
learned embedding of the first seed is saved, so that the difficulty profile can be
computed in the representation each method learns.

Install:  pip install clustpy torch
Usage:    python run_deep_clustering.py --data_dir datasets/suite [--resume]
          python run_deep_clustering.py --demo          # synthetic smoke test
Datasets: .npz files with X (n, d) and y (n,), or .csv files with feature columns
          prefixed 'f' and a 'label' column (see load_dataset()).
Output:   results/deep_results.csv          one row per (dataset, method, run)
          results/deep_labels.npz           every label vector, key "dataset|method|seed"
          results/deep_embeddings_seed42.npz embedding learned with the first seed,
                                            key "dataset|method" (n x embedding size)
          results/deep_run_info.csv         versions and platform of the run
          results/deep_results_summary.csv  mean / std per (dataset, method)
CPU is sufficient for n = 1000, d = 20; see CLUSTPY_DEVICE below for a GPU.
"""

import os
import sys
import glob
import time
import argparse
import warnings
import contextlib
import io

import platform

import numpy as np
import pandas as pd

# --- Device (CPU by default). For a GPU, uncomment:
# os.environ['CLUSTPY_DEVICE'] = 'cuda:0'

# Silence the tqdm progress bars of the autoencoder training, for a readable log.
os.environ["TQDM_DISABLE"] = "1"

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

# The evaluation of the classical tier, so that both tiers use the same conventions
from clustaudit.audit.evaluation import evaluate_partition, result_row, RESULT_COLUMNS

# ClustPy: reference implementations of the three deep methods
from clustpy.deep import IDEC, DCN, DipDECK


# ============================================================================
# 1. GLOBAL PARAMETERS
# ============================================================================
SEEDS = [42, 123, 456, 789, 1011]   # five runs per dataset, as in the paper
N_TRUE_CLUSTERS = 5                  # K vrai de la suite maison (paper: K = 5)
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(_RESULTS_DIR, "deep_results.csv")
LABELS_NPZ = os.path.join(_RESULTS_DIR, "deep_labels.npz")
INFO_CSV = os.path.join(_RESULTS_DIR, "deep_run_info.csv")
EMBED_NPZ = os.path.join(_RESULTS_DIR, "deep_embeddings_seed42.npz")
SUMMARY_CSV = os.path.join(_RESULTS_DIR, "deep_results_summary.csv")


# ============================================================================
# 2. METRICS  (always computed in the original space X)
# ============================================================================
def compute_metrics(X, y_true, y_pred):
    """Evaluation of one partition, identical to the classical tier's: inlier ARI/NMI
    (ARI), noise as a class (ARI_full), kept points (ARI_kept), internal indices on the
    kept points, number of clusters found, smallest-cluster F1, failure mode."""
    return result_row(evaluate_partition(X, y_true, y_pred))


# ============================================================================
# 3. THE THREE DEEP METHODS  (common interface: fit -> labels_)
# ============================================================================
def build_model(method, seed):
    """Instantiate one of the three methods with its default configuration (fixed protocol).
    DipDECK estime K : on ne lui impose pas N_TRUE_CLUSTERS."""
    if method == "IDEC":
        return IDEC(n_clusters=N_TRUE_CLUSTERS, random_state=seed)
    if method == "DCN":
        return DCN(n_clusters=N_TRUE_CLUSTERS, random_state=seed)
    if method == "DipDECK":
        # n_clusters_init = surestimation initiale ; l'algo fusionne ensuite.
        return DipDECK(random_state=seed)
    raise ValueError(f"Unknown method: {method}")


def run_one(method, X, y_true, seed, embed=False):
    """Run one method once; return (metrics, labels, embedding or None)."""
    model = build_model(method, seed)
    t0 = time.time()
    # DipDECK is verbose: its stdout is captured, for a readable log.
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X)
    runtime = time.time() - t0
    y_pred = np.asarray(model.labels_).astype(int)
    metrics = compute_metrics(X, y_true, y_pred)
    metrics["runtime_s"] = round(runtime, 2)
    Z = None
    if embed:
        with contextlib.redirect_stdout(io.StringIO()):
            Z = np.asarray(model.transform(X), dtype=np.float32)
    return metrics, y_pred, Z


# ============================================================================
# 4. DATA LOADING
# ============================================================================
def load_dataset(path):
    """Load a dataset from .npz (keys X, y) or .csv (feature columns 'f*' and a
    'label' column). Adapt here for another file format."""
    if path.endswith(".npz"):
        d = np.load(path)
        return d["X"], d["y"]
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        label_col = "label" if "label" in df.columns else df.columns[-1]
        feat_cols = [c for c in df.columns if c != label_col]
        return df[feat_cols].to_numpy(), df[label_col].to_numpy()
    raise ValueError(f"Unsupported format: {path}")


def iter_datasets(data_dir):
    """Iterate over the datasets of a folder; yield (name, X, y)."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:   # csv datasets only when there is no npz (a suite folder holds manifest.csv)
        paths = sorted(p for p in glob.glob(os.path.join(data_dir, "*.csv"))
                       if not os.path.basename(p).startswith("manifest"))
    if not paths:
        print(f"[!] No .npz/.csv found in {data_dir}")
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        X, y = load_dataset(p)
        yield name, X, y


def iter_demo_datasets():
    """Demo mode: six synthetic datasets of increasing difficulty (decreasing
    separation), to check the pipeline without external data."""
    from sklearn.datasets import make_blobs
    for i, std in enumerate([1.0, 2.0, 3.0, 4.0, 6.0, 8.0]):
        X, y = make_blobs(n_samples=1000, n_features=20,
                          centers=N_TRUE_CLUSTERS, cluster_std=std,
                          random_state=0)
        yield f"demo_std{std}", X, y


def write_run_info(seeds, args):
    """Versions and platform of the run, for the provenance of the result files."""
    import sklearn
    import torch
    import clustpy
    info = [("script", "benchmarks/deep/run_deep_clustering.py"),
            ("python", platform.python_version()), ("platform", platform.platform()),
            ("numpy", np.__version__), ("pandas", pd.__version__),
            ("scikit-learn", sklearn.__version__), ("torch", torch.__version__),
            ("clustpy", getattr(clustpy, "__version__", "?")),
            ("seeds", " ".join(str(s) for s in seeds)),
            ("data_dir", "demo" if args.demo else os.path.relpath(args.data_dir)),
            ("started", time.strftime("%Y-%m-%d %H:%M:%S"))]
    pd.DataFrame(info, columns=["key", "value"]).to_csv(INFO_CSV, index=False)


# ============================================================================
# 5. MAIN LOOP
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Deep tier of the audit (IDEC, DCN, DipDECK).")
    ap.add_argument("--data_dir", type=str, default=None,
                    help="Dossier de datasets (.npz ou .csv).")
    ap.add_argument("--demo", action="store_true",
                    help="Demo mode (synthetic data).")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help=f"Number of seeds (default {len(SEEDS)}, as in the paper).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume: skip datasets already present in the results CSV.")
    args = ap.parse_args()

    if not args.demo and args.data_dir is None:
        print("Give --data_dir <folder>  or  --demo")
        sys.exit(1)

    seeds = SEEDS[:args.seeds] if args.seeds <= len(SEEDS) else \
        list(range(args.seeds))
    methods = ["IDEC", "DCN", "DipDECK"]
    source = iter_demo_datasets() if args.demo else iter_datasets(args.data_dir)

    rows, labels, embeddings = [], {}, {}
    done = set()
    if args.resume and os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        if "ARI_full" not in prev.columns:
            print(f"[!] {RESULTS_CSV} was written by an older version of this script "
                  "(no ARI_full column). Move it aside and run without --resume.")
            sys.exit(1)
        rows = prev.to_dict("records")
        done = set(prev["dataset"].unique())
        if os.path.exists(LABELS_NPZ):
            with np.load(LABELS_NPZ) as z:
                labels = {k: z[k] for k in z.files}
        if os.path.exists(EMBED_NPZ):
            with np.load(EMBED_NPZ) as z:
                embeddings = {k: z[k] for k in z.files}
        print(f"[resume] {len(done)} datasets already in {RESULTS_CSV}")
    write_run_info(seeds, args)
    for name, X, y in source:
        if name in done:
            continue
        # Standardization: autoencoders are sensitive to scale. The same
        # normalization is applied for every method, for comparability.
        Xs = StandardScaler().fit_transform(X.astype(np.float64))
        y = np.asarray(y)
        for method in methods:
            for seed in seeds:
                try:
                    m, pred, Z = run_one(method, Xs, y, seed, embed=(seed == seeds[0]))
                    labels[f"{name}|{method}|{seed}"] = pred.astype(np.int16)
                    if Z is not None:
                        embeddings[f"{name}|{method}"] = Z
                    status = "ok"
                except Exception as e:
                    m = {c: np.nan for c in RESULT_COLUMNS}
                    m["failure_mode"] = "error"
                    m["runtime_s"] = np.nan
                    status = f"ERROR: {type(e).__name__}: {e}"
                row = {"dataset": name, "method": method, "seed": seed,
                       "status": status, **m}
                rows.append(row)
                print(f"[{name:18s}] {method:8s} seed={seed:<5d} "
                      f"ARI={float(m['ARI']):.3f} ARI_full={float(m['ARI_full']):.3f} "
                      f"K={m['n_clusters_found']} {status if status!='ok' else ''}", flush=True)
        # incremental write: a crash never loses more than one dataset
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        np.savez_compressed(LABELS_NPZ, **labels)
        np.savez_compressed(EMBED_NPZ, **embeddings)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    np.savez_compressed(LABELS_NPZ, **labels)
    np.savez_compressed(EMBED_NPZ, **embeddings)
    print(f"\n[OK] Detailed results -> {RESULTS_CSV}  ({len(df)} rows)")

    # Summary: mean and std per (dataset, method)
    agg = (df.groupby(["dataset", "method"])
             .agg(ARI_mean=("ARI", "mean"), ARI_std=("ARI", "std"),
                  ARI_full_mean=("ARI_full", "mean"),
                  NMI_mean=("NMI", "mean"), NMI_std=("NMI", "std"),
                  Sil_mean=("Silhouette", "mean"),
                  K_found_mean=("n_clusters_found", "mean"),
                  runtime_mean=("runtime_s", "mean"))
             .round(3).reset_index())
    agg.to_csv(SUMMARY_CSV, index=False)
    print(f"[OK] Summary          -> {SUMMARY_CSV}")
    print("\n--- Summary preview ---")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
