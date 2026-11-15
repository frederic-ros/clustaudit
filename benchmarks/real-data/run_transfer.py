#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transfer of the difficulty profile to real data (Section 6.4 of the paper).

Nineteen UCI / OpenML datasets, labels taken as given (or, with --relabel, made
consistent with the k-NN rule of the pre-noise refinement). The difficulty profile
is computed with the same instrument as on generated data. The eight classical
methods are run, and two label-free quantities are recorded per dataset: the median
Silhouette of the non-degenerate partitions (clean_median) and the fraction of
degenerate outputs. The ARI is not used as the criterion, since real labels are
semantic. Other aggregates (mean, mean - std, median, 20% trimmed mean, clean_mean)
are written for completeness; clean_median is the one the paper reports.

Usage:   python run_transfer.py [--relabel]
Output:  results/transfer_results[_relabel].csv, results/transfer_per_algo[_relabel].csv,
         results/transfer_plot[_relabel].png
Data are fetched from scikit-learn and OpenML on first use (network access needed).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, trim_mean

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import (load_iris, load_wine, load_breast_cancer,
                              load_digits, fetch_openml)
from sklearn.cluster import (KMeans, SpectralClustering,
                             AgglomerativeClustering, DBSCAN, OPTICS)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
try:
    from sklearn.cluster import HDBSCAN
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

# The difficulty profile comes from the package itself: no more stand-alone copy.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
try:
    import clustaudit  # noqa: F401
except ImportError:
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, _os.pardir, _os.pardir)))
_RESULTS_DIR = _os.path.join(_HERE, "results")
_os.makedirs(_RESULTS_DIR, exist_ok=True)
from clustaudit.metrics import difficulty_profile

# Real labels are taken as given by default; --relabel applies the pre-noise refinement first.
RELABEL = "--relabel" in _sys.argv


# ============================================================================
#  REAL DATASETS
# ============================================================================
def load_real_datasets(subsample=2000):
    out = []
    for name, loader in [("iris", load_iris), ("wine", load_wine),
                         ("breast_cancer", load_breast_cancer),
                         ("digits", load_digits)]:
        d = loader()
        out.append((name, d.data.astype(float), d.target.astype(int)))
    openml_specs = ["seeds", "ecoli", "glass", "vehicle", "yeast",
                    "ionosphere", "sonar", "heart-statlog", "diabetes",
                    "vowel", "waveform-5000", "satimage", "optdigits",
                    "pendigits", "banknote-authentication"]
    rng = np.random.RandomState(0)
    for oml in openml_specs:
        try:
            ds = fetch_openml(name=oml, version=1, as_frame=True)
            X = ds.data.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            _, y = np.unique(ds.target.to_numpy(), return_inverse=True)
            y = y.astype(int)
            ok = ~np.isnan(X).any(axis=1)
            X, y = X[ok], y[ok]
            if X.shape[0] > subsample:
                idx = rng.choice(X.shape[0], subsample, replace=False)
                X, y = X[idx], y[idx]
            out.append((oml.split("-")[0], X, y))
        except Exception as e:
            print(f"[!] OpenML '{oml}' skipped ({type(e).__name__}).")
    return [(nm, StandardScaler().fit_transform(X), y) for nm, X, y in out]


# ============================================================================
#  ALGORITHMS AND PER-ALGORITHM SILHOUETTE
# ============================================================================
def per_algorithm_silhouettes(X, K):
    def run(name, fn):
        try:
            lab = np.asarray(fn())
        except Exception as e:   # recorded, not dropped: excluded from the aggregates (NaN Silhouette)
            print(f"  ! {name} failed: {type(e).__name__}: {e} (recorded)")
            return {"algo": name, "sil": np.nan, "degenerate": True, "noise_frac": np.nan,
                    "status": f"error: {type(e).__name__}: {e}"[:300]}
        mask = lab != -1
        noise_frac = 1.0 - mask.mean()
        uniq = set(lab[mask].tolist()) if mask.any() else set()
        degenerate = (len(uniq) < 2) or (noise_frac > 0.5)
        sil = np.nan
        if len(uniq) >= 2 and mask.sum() > len(uniq):
            try:
                sil = silhouette_score(X[mask], lab[mask])
            except Exception:
                sil = np.nan
        return {"algo": name, "sil": sil, "degenerate": degenerate,
                "noise_frac": round(noise_frac, 3), "status": "ok"}

    specs = [
        ("KMeans",   lambda: KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X)),
        ("GMM",      lambda: GaussianMixture(n_components=K, random_state=0).fit_predict(X)),
        ("Ward",     lambda: AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)),
        ("Average",  lambda: AgglomerativeClustering(n_clusters=K, linkage="average").fit_predict(X)),
        ("Spectral", lambda: SpectralClustering(n_clusters=K, affinity="nearest_neighbors",
                                                random_state=0).fit_predict(X)),
        ("DBSCAN",   lambda: DBSCAN().fit_predict(X)),
        ("OPTICS",   lambda: OPTICS(min_samples=10).fit_predict(X)),
    ]
    if _HAS_HDBSCAN:
        specs.append(("HDBSCAN", lambda: HDBSCAN(min_cluster_size=10).fit_predict(X)))
    return [run(n, f) for n, f in specs]


def aggregates(recs):
    sils = np.array([r["sil"] for r in recs if not np.isnan(r["sil"])])
    clean = np.array([r["sil"] for r in recs
                      if (not np.isnan(r["sil"])) and (not r["degenerate"])])
    o = {"n_algos_total": len(sils), "n_algos_clean": len(clean)}
    if len(sils) == 0:
        o.update({k: np.nan for k in
                  ["mean", "std", "Q", "median", "trimmed20",
                   "clean_mean", "clean_median"]})
        return o
    o["mean"] = float(np.mean(sils)); o["std"] = float(np.std(sils))
    o["Q"] = o["mean"] - o["std"]
    o["median"] = float(np.median(sils))
    o["trimmed20"] = float(trim_mean(sils, 0.2)) if len(sils) >= 5 else o["mean"]
    o["clean_mean"] = float(np.mean(clean)) if len(clean) else np.nan
    o["clean_median"] = float(np.median(clean)) if len(clean) else np.nan
    return o


# Label offsets (dx, dy in points) to avoid overlaps
# between close pairs (sonar/vehicle, optdigits/digits, ecoli/wine...).
_LABEL_OFFSETS = {
    "sonar": (5, -11), "vehicle": (5, 5),
    "optdigits": (5, -11), "digits": (5, 5),
    "ecoli": (-8, 7), "satimage": (5, 5),
    "breast_cancer": (-30, 9), "wine": (6, -10),
    "diabetes": (5, -11), "banknote": (5, 5),
}


# ============================================================================
#  MAIN
# ============================================================================
def main():
    datasets = load_real_datasets()
    rows, dump = [], []
    for name, X, y in datasets:
        K = len(np.unique(y))
        prof = difficulty_profile(X, y, relabel=RELABEL)
        D = prof.score
        prof_cols = {k: v for k, v in prof.to_dict().items()
                     if k not in ("D_dataset", "n_samples", "dimension", "n_clusters")}
        recs = per_algorithm_silhouettes(X, K)
        agg = aggregates(recs)
        rows.append({"dataset": name, "n": X.shape[0], "d": X.shape[1], "K": K,
                     "D_dataset": round(D, 4) if not np.isnan(D) else np.nan,
                     "relabel": RELABEL, **prof_cols,
                     **{k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in agg.items()}})
        for r in recs:
            dump.append({"dataset": name, **r})
        ndeg = sum(r["degenerate"] for r in recs)
        cm = agg["clean_median"]
        print(f"[{name:14s}] D={D:.3f}  mean={agg['mean']:.3f} "
              f"median={agg['median']:.3f} "
              f"clean_med={cm if not np.isnan(cm) else float('nan'):.3f} "
              f"({ndeg}/{len(recs)} degenerate)")

    df = pd.DataFrame(rows)
    sfx = "_relabel" if RELABEL else ""   # --relabel must not overwrite the labels-as-given results
    df.to_csv(_os.path.join(_RESULTS_DIR, f"transfer_results{sfx}.csv"), index=False)
    pd.DataFrame(dump).to_csv(_os.path.join(_RESULTS_DIR, f"transfer_per_algo{sfx}.csv"), index=False)
    print("\n[OK] -> transfer_results.csv & transfer_per_algo.csv")

    def partial_K(x, y, K):
        def resid(a, b):
            A = np.vstack([b, np.ones_like(b)]).T
            c, _, _, _ = np.linalg.lstsq(A, a, rcond=None)
            return a - A @ c
        return spearmanr(resid(x, K), resid(y, K))

    sub = df.dropna(subset=["D_dataset"])
    Karr = sub["K"].to_numpy(float); D = sub["D_dataset"].to_numpy()
    print("\n=== D_dataset vs aggregates (rho | partial rho given K) ===")
    print("    Aggregate reported in the paper: clean_median (excludes")
    print("    degenerate partitions).")
    for col in ["mean", "Q", "median", "trimmed20", "clean_mean", "clean_median"]:
        y = sub[col].to_numpy(); m = ~np.isnan(y)
        if m.sum() < 4:
            continue
        rho, p = spearmanr(D[m], y[m])
        rhop, pp = partial_K(D[m], y[m], Karr[m])
        star = "  <== reported in the paper" if col == "clean_median" else ""
        print(f"  {col:13s} rho={rho:+.3f}(p={p:.3f}) | "
              f"partial|K rho={rhop:+.3f}(p={pp:.3f}){star}")

    # ------------------------------------------------------------------
    # Quick-look figure: D_dataset vs clean_median (degenerate partitions excluded).
    # Bars: standard deviation of the Silhouette over the non-degenerate methods.
    # The paper's Figure 10 is drawn by analysis/make_paper_figures_v2.py.
    # ------------------------------------------------------------------
    s = df.dropna(subset=["D_dataset", "clean_median"]).copy()
    rho, p = spearmanr(s["D_dataset"], s["clean_median"])
    # standard deviation over the non-degenerate methods, per dataset
    clean_std = []
    per = pd.DataFrame(dump)
    for _, r in s.iterrows():
        sub_algo = per[(per["dataset"] == r["dataset"]) &
                       (~per["degenerate"]) & (per["sil"].notna())]
        clean_std.append(sub_algo["sil"].std() if len(sub_algo) > 1 else 0.0)
    s["clean_std"] = clean_std

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.errorbar(s["D_dataset"], s["clean_median"], yerr=s["clean_std"], fmt="o",
                ms=6.5, capsize=3, color="#2b6cb0", ecolor="#cbd5e0",
                elinewidth=1.4,
                label="clean median $\\pm$ std (degenerate methods excluded)",
                zorder=3)
    coef = np.polyfit(s["D_dataset"], s["clean_median"], 1)
    xs = np.linspace(s["D_dataset"].min(), s["D_dataset"].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), "--", color="#e53e3e", lw=1.3,
            label=f"trend ($\\rho={rho:.2f}$, $p={p:.3f}$)", zorder=2)
    for _, r in s.iterrows():
        dx, dy = _LABEL_OFFSETS.get(r["dataset"], (5, 3))
        ax.annotate(r["dataset"], (r["D_dataset"], r["clean_median"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=6.8)
    ax.set_xlabel(r"$D_{\mathrm{dataset}}$ (intrinsic difficulty, reconstructed from labels)")
    ax.set_ylabel("Robust geometric quality (clean median)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.08)
    fig.tight_layout(); fig.savefig(_os.path.join(_RESULTS_DIR, "transfer_plot" + ("_relabel" if RELABEL else "") + ".png"), dpi=160)
    print(f"\n[OK] -> transfer_plot (y axis = clean_median, rho={rho:+.3f}, p={p:.3f})")


if __name__ == "__main__":
    main()
