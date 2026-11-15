#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep tier of the audit: IDEC, DCN (K given) and DipDECK (K estimated), via ClustPy.

For every dataset of a folder (or a small synthetic demo set) the script runs the
three methods with their reference defaults, five runs each, and records ARI, NMI,
Silhouette, Calinski-Harabasz, Davies-Bouldin and the number of clusters found.
Each method is judged on its label vector only (black box); internal indices are
computed in the original space X, never in the learned embedding, so that they are
comparable with the classical tier.

Install:  pip install clustpy torch
Usage:    python run_deep_clustering.py --data_dir datasets/suite [--resume]
          python run_deep_clustering.py --demo          # synthetic smoke test
Datasets: .npz files with X (n, d) and y (n,), or .csv files with feature columns
          prefixed 'f' and a 'label' column (see load_dataset()).
Output:   results/deep_results.csv          one row per (dataset, method, run)
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

import numpy as np
import pandas as pd

# --- Device (CPU by default). For a GPU, uncomment:
# os.environ['CLUSTPY_DEVICE'] = 'cuda:0'

# Silence the tqdm progress bars of the autoencoder training, for a readable log.
os.environ["TQDM_DISABLE"] = "1"

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

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
SUMMARY_CSV = os.path.join(_RESULTS_DIR, "deep_results_summary.csv")


# ============================================================================
# 2. METRICS  (always computed in the original space X)
# ============================================================================
def compute_metrics(X, y_true, y_pred):
    """External and internal metrics of one partition.

    Internal indices are defined only if at least 2 clusters are predicted;
    NaN otherwise (same convention as the classical tier)."""
    out = {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "n_clusters_found": len(set(y_pred) - {-1}),  # -1 = rejected points, if any
    }
    # Internal indices need >= 2 clusters and not only noise
    mask = y_pred != -1
    uniq = set(y_pred[mask]) if mask.any() else set()
    if len(uniq) >= 2 and mask.sum() > len(uniq):
        try:
            out["Silhouette"] = silhouette_score(X[mask], y_pred[mask])
            out["CH"] = calinski_harabasz_score(X[mask], y_pred[mask])
            out["DB"] = davies_bouldin_score(X[mask], y_pred[mask])
        except Exception:
            out["Silhouette"] = out["CH"] = out["DB"] = np.nan
    else:
        out["Silhouette"] = out["CH"] = out["DB"] = np.nan
    return out


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


def run_one(method, X, y_true, seed):
    """Run one method once; return (metrics, runtime)."""
    model = build_model(method, seed)
    t0 = time.time()
    # DipDECK is verbose: its stdout is captured, for a readable log.
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X)
    runtime = time.time() - t0
    y_pred = np.asarray(model.labels_)
    metrics = compute_metrics(X, y_true, y_pred)
    metrics["runtime_s"] = round(runtime, 2)
    return metrics


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

    rows = []
    done = set()
    if args.resume and os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        rows = prev.to_dict("records")
        done = set(prev["dataset"].unique())
        print(f"[resume] {len(done)} datasets already in {RESULTS_CSV}")
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
                    m = run_one(method, Xs, y, seed)
                    status = "ok"
                except Exception as e:
                    m = {"ARI": np.nan, "NMI": np.nan, "Silhouette": np.nan,
                         "CH": np.nan, "DB": np.nan, "n_clusters_found": np.nan,
                         "runtime_s": np.nan}
                    status = f"ERROR: {type(e).__name__}: {e}"
                row = {"dataset": name, "method": method, "seed": seed,
                       "status": status, **m}
                rows.append(row)
                print(f"[{name:18s}] {method:8s} seed={seed:<5d} "
                      f"ARI={m['ARI'] if isinstance(m['ARI'],float) else float('nan'):.3f} "
                      f"K={m['n_clusters_found']} {status if status!='ok' else ''}")
        # incremental write: a crash never loses more than one dataset
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[OK] Detailed results -> {RESULTS_CSV}  ({len(df)} rows)")

    # Summary: mean and std per (dataset, method)
    agg = (df.groupby(["dataset", "method"])
             .agg(ARI_mean=("ARI", "mean"), ARI_std=("ARI", "std"),
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
