#!/usr/bin/env python
"""
Q1 -- Ablation: does the profile explain degradation beyond simple baselines?

Reads every result CSV produced by the benchmarks (exp1, exp2, exp5 and any
sweep CSV you pass), aggregates to the dataset level (mean over the algorithm
pool, or per family), and reports the leave-one-out R^2 of mean ARI explained
by each predictor set, plus Pearson/Spearman with bootstrap CIs.

    python analysis/q1_ablation.py                       # default CSVs
    python analysis/q1_ablation.py --csv a.csv b.csv     # explicit
    python analysis/q1_ablation.py --target Silhouette   # another target

Output: analysis/tables/q1_ablation.csv (+ .tex) and a console summary.
"""
from __future__ import annotations
import argparse, glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_CSVS = [
    "benchmarks/classical/results/exp1_difficulty_validation/exp1_results*.csv",
    "benchmarks/classical/results/exp2_sensitivity_analysis/exp2_results*.csv",
    "benchmarks/classical/results/exp5_high_dimensional_case_study/exp5_results*.csv",
    "benchmarks/classical/results/sweeps_v2/sweep_*.csv",
]
FAMILY = {"KMeans": "centroid", "GMM": "centroid", "Spectral": "graph",
          "Ward": "connectivity", "Average": "connectivity", "Complete": "connectivity",
          "DBSCAN": "density", "HDBSCAN": "density", "OPTICS": "density"}
BASE = ["base_one_minus_silhouette", "base_knn_cv_error", "base_within_between_ratio"]
COMP = ["S_separation", "S_boundary", "S_shape", "S_noise"]
KNN = ["knn_edge_impurity", "knn_ambiguous_fraction", "knn_mean_impurity", "knn_fragmentation",
       "knn_density_heterogeneity", "knn_intra_density_spread", "knn_noise_fraction", "knn_noise_proximity"]
SETS = {
    "1-Silhouette (reference)": ["base_one_minus_silhouette"],
    "kNN CV error": ["base_knn_cv_error"],
    "within/between ratio": ["base_within_between_ratio"],
    "all 3 baselines": BASE,
    "S_sep": ["S_separation"], "S_bnd": ["S_boundary"], "S_shape": ["S_shape"], "S_noise": ["S_noise"],
    "D_dataset (scalar)": ["D_dataset"],
    "4 components": COMP,
    "kNN descriptors": KNN,
    "components + kNN": COMP + KNN,
    "full profile (+baselines)": COMP + KNN + BASE,
}


def dataset_key(df):
    """One id per generated dataset: whatever identifies it in each CSV."""
    for cols in (["config_id", "seed"], ["parameter", "value", "value_eta", "value_eta_int", "seed"],
                 ["dimension", "seed"], ["dataset"]):
        if all(c in df.columns for c in cols):
            return df[cols].fillna("na").astype(str).agg("|".join, axis=1)
    return df.index.astype(str)


def loo_r2(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[ok], y[ok]
    if len(y) < 8:
        return np.nan, len(y)
    model = RidgeCV(alphas=np.logspace(-3, 2, 12))
    pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), len(y)


def boot_ci(x, y, fn, n=1000, seed=0):
    rng = np.random.RandomState(seed); x = np.asarray(x); y = np.asarray(y)
    ok = ~np.isnan(x) & ~np.isnan(y); x, y = x[ok], y[ok]
    if len(x) < 5:
        return np.nan, (np.nan, np.nan)
    stat = fn(x, y)[0]
    bs = [fn(x[i], y[i])[0] for i in (rng.randint(0, len(x), len(x)) for _ in range(n))]
    return stat, tuple(np.nanpercentile(bs, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*")
    ap.add_argument("--target", default="ARI")
    ap.add_argument("--criterion", default=None, help="keep only rows with this tuning criterion")
    a = ap.parse_args()
    files = a.csv or [f for pat in DEFAULT_CSVS for f in glob.glob(os.path.join(ROOT, pat))]
    if not files:
        sys.exit("no result CSV found; run the benchmarks first")
    frames = []
    for f in files:
        df = pd.read_csv(f, sep=None, engine="python")
        if "knn_edge_impurity" not in df.columns:
            print(f"[skip] {f}: v1 CSV without profile columns"); continue
        if a.criterion and "criterion" in df.columns:
            df = df[df["criterion"] == a.criterion]
        df["dataset_key"] = os.path.basename(f) + "::" + dataset_key(df)
        if "knn_noise_proximity" in df.columns:   # pre-2.1.7 CSVs: NaN on noise-free datasets, now 0
            df["knn_noise_proximity"] = df["knn_noise_proximity"].fillna(0.0)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["family"] = df["algorithm"].map(FAMILY).fillna("other")
    preds = sorted(set(sum(SETS.values(), [])))
    preds = [p for p in preds if p in df.columns]

    rows, corr_rows = [], []
    groups = {"all": df} | {f: g for f, g in df.groupby("family")}
    for gname, g in groups.items():
        agg = g.groupby("dataset_key").agg({a.target: "mean", **{p: "first" for p in preds}}).dropna(subset=[a.target])
        for sname, cols in SETS.items():
            cols = [c for c in cols if c in agg.columns]
            if not cols: continue
            r2, n = loo_r2(agg[cols], agg[a.target])
            rows.append({"group": gname, "predictor_set": sname, "loo_R2": round(r2, 3), "N": n})
        for p in preds:
            r, rci = boot_ci(agg[p], agg[a.target], pearsonr)
            rho, rhoci = boot_ci(agg[p], agg[a.target], spearmanr)
            corr_rows.append({"group": gname, "predictor": p, "pearson": round(r, 3),
                              "r_ci_lo": round(rci[0], 3), "r_ci_hi": round(rci[1], 3),
                              "spearman": round(rho, 3), "rho_ci_lo": round(rhoci[0], 3), "rho_ci_hi": round(rhoci[1], 3)})

    out = os.path.join(HERE, "tables"); os.makedirs(out, exist_ok=True)
    tab = pd.DataFrame(rows).pivot(index="predictor_set", columns="group", values="loo_R2").reindex(list(SETS))
    tab.to_csv(os.path.join(out, f"q1_ablation_{a.target}.csv"))
    with open(os.path.join(out, f"q1_ablation_{a.target}.tex"), "w") as f:
        f.write(tab.to_latex(float_format="%.3f", na_rep="--"))
    pd.DataFrame(corr_rows).to_csv(os.path.join(out, f"q1_correlations_{a.target}.csv"), index=False)
    print(f"\nLOO R^2 of mean {a.target} (dataset level) by predictor set\n")
    print(tab.to_string())
    print(f"\n-> {out}/q1_ablation_{a.target}.csv / .tex ; q1_correlations_{a.target}.csv (bootstrap CIs)")


if __name__ == "__main__":
    main()
