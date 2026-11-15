#!/usr/bin/env python
"""The new single-factor sweeps of the v2 paper (overlap, imbalance, inter-density, intra-density,
connectivity), over the classical pool under protocol S. Writes one CSV per factor into
benchmarks/classical/results/sweeps_v2/. Edit VALUES / SEEDS / --criterion as needed.

    python scripts/sweeps_v2.py                 # all factors
    python scripts/sweeps_v2.py --factor overlap --seeds 42 123
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from clustaudit.audit import sweep, run_method, CLASSICAL_EIGHT

VALUES = {
    "overlap":       [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
    "imbalance":     [1.0, 1.5, 2.0, 3.0, 5.0],
    "inter_density": [0.0, 0.25, 0.5, 0.75, 1.0],
    "intra_density": [0.0, 0.5, 1.0, 2.0],
    "connectivity":  [0.0, 0.25, 0.5, 0.75, 1.0],
    "eta_int":       [0.0, 0.25, 0.5, 0.75, 1.0],
    "eta":           [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "anisotropy":    [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                   "benchmarks", "classical", "results", "sweeps_v2")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", nargs="*", default=list(VALUES))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456, 789, 1011])
    ap.add_argument("--criterion", default="silhouette", choices=["silhouette", "dbcv", "fixed"])
    ap.add_argument("--algos", nargs="*", default=None, help="subset of the pool (e.g. --algos DPC); output gets a suffix")
    ap.add_argument("--n", type=int, default=1000, help="number of samples (control for absolute cluster sizes)")
    ap.add_argument("--base-eta", type=float, default=None, help="override the base noise ratio (e.g. 0.3 for the eta_int crossover)")
    ap.add_argument("--base-eta-int", type=float, default=None)
    ap.add_argument("--K", type=int, default=5, help="number of clusters (control for K = 5 everywhere)")
    ap.add_argument("--no-prerefine", action="store_true", help="skip the pre-noise label refinement (raw generative labels as reference)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pool = a.algos or CLASSICAL_EIGHT + ["DPC"]
    # run_method returns (labels, status); audit_dataset records every run, including
    # tunings without an admissible configuration and methods that raise
    algos = {n: (lambda name: (lambda X, seed: run_method(name, X, seed, a.criterion, n_clusters=a.K)))(n)
             for n in pool}
    suffix = ("_" + "_".join(a.algos)) if a.algos else ""
    if a.n != 1000: suffix += f"_n{a.n}"
    if a.base_eta is not None: suffix += f"_eta{a.base_eta}"
    if a.no_prerefine: suffix += "_noprerefine"
    if a.K != 5: suffix += f"_K{a.K}"
    for f in a.factor:
        print(f"== sweep {f}: {VALUES[f]} x seeds {a.seeds}")
        from clustaudit.audit.runner import _base_config
        base = _base_config()
        if a.base_eta is not None: base["noise"].noise_ratio = a.base_eta
        if a.base_eta_int is not None: base["noise"].internal_noise_ratio = a.base_eta_int
        if a.no_prerefine: base["ambiguity"].enable_pre_refinement = False
        df = sweep(algos, factor=f, values=VALUES[f], seeds=a.seeds, n_samples=a.n, n_clusters=a.K, base=base)
        df["criterion"] = a.criterion
        path = os.path.join(OUT, f"sweep_{f}_{a.criterion}{suffix}.csv")
        df.to_csv(path, index=False)
        print(df.groupby(["value"])[["ARI", "D_dataset", "knn_edge_impurity"]].mean().round(3))
        print(f"-> {path}")

if __name__ == "__main__":
    main()
