#!/usr/bin/env python
"""
Q2 -- How much does the measurement instrument (Theta_amb) matter?

Recomputes the difficulty profile of every dataset of a suite under perturbed
instrument settings and reports (i) the Spearman correlation of the dataset
ordering with the default instrument, for D_dataset and for each descriptor,
and (ii) the correlation of each perturbed D_dataset with mean ARI if an
exp1 result CSV is available (so the headline correlation gets a range).

    python analysis/q2_instrument_sensitivity.py --suite datasets/suite
    python analysis/q2_instrument_sensitivity.py --suite datasets/suite --draws 200

Output: analysis/tables/q2_instrument_sensitivity.csv (+ q2_weights_sensitivity.csv)
"""
from __future__ import annotations
import argparse, glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

from clustaudit import AmbiguityParams, difficulty_profile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))

PERTURBATIONS = {
    "default":            dict(),
    "knn k=5":            dict(pre_refinement_k_neighbors=5),
    "knn k=2sqrt(n)":     dict(pre_refinement_k_neighbors=63),
    "random forest":      dict(post_refinement_classifier="random_forest"),
    "svm":                dict(post_refinement_classifier="svm"),
    "tau_core=0.90":      dict(core_threshold=0.90),
    "tau_core=0.99":      dict(core_threshold=0.99),
    "q_gap=0.05":         dict(distant_noise_quantile=0.05),
    "q_gap=0.25":         dict(distant_noise_quantile=0.25),
    "in-sample confidence": dict(confidence_mode="in_sample"),
    "folds=10":           dict(cv_folds=10),
}
COMP = ["separation", "boundary", "shape", "noise"]


def rescore(components, w, consts=None):
    """D with perturbed weights (components already normalised)."""
    w = np.asarray(w) / np.sum(w)
    return float(np.dot(w, [components[c] for c in COMP]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=os.path.join(ROOT, "datasets", "suite"))
    ap.add_argument("--exp1", default=os.path.join(ROOT, "benchmarks/classical/results/exp1_difficulty_validation/exp1_results_fixed.csv"),
                    help="suite audit whose mean ARI gives r(D, ARI) (default: protocol F, as in the paper)")
    ap.add_argument("--draws", type=int, default=200, help="random weight perturbations (+/-30%%)")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.suite, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz in {a.suite}; run `clustaudit suite` first")
    data = [(os.path.basename(f), np.load(f)) for f in files]

    # mean ARI per dataset, if exp1 is there (matched on seed+config order = file order)
    ari = None
    if os.path.exists(a.exp1):
        e = pd.read_csv(a.exp1, float_precision="round_trip")
        if {"config_id", "seed", "ARI"} <= set(e.columns):
            if "algorithm" in e:   # the paper's suite pool: DPC is reported separately
                e = e[e.algorithm != "DPC"]
            m = e.groupby(["config_id", "seed"])["ARI"].mean().reset_index()
            m["dataset"] = [f"dataset_{i:03d}" for i in range(len(m))]
            ari = dict(zip(m["dataset"], m["ARI"]))

    profiles = {}
    for name, kw in PERTURBATIONS.items():
        amb = AmbiguityParams(**kw)
        rows = []
        for fname, z in data:
            try:
                p = difficulty_profile(z["X"], z["y"], ambiguity=amb, k=a.k, with_baselines=False)
                rows.append({"dataset": fname, "D": p.score, **{f"S_{c}": p.components[c] for c in COMP},
                             **{f"knn_{k}": v for k, v in p.knn.items()}})
            except Exception as ex:
                print(f"  ! {name} / {fname}: {ex}")
        profiles[name] = pd.DataFrame(rows).set_index("dataset")
        print(f"[{name}] done ({len(rows)} datasets)")

    ref = profiles["default"]
    out_rows = []
    for name, df in profiles.items():
        common = ref.index.intersection(df.index)
        row = {"perturbation": name, "N": len(common)}
        for col in ["D"] + [f"S_{c}" for c in COMP] + ["knn_edge_impurity", "knn_fragmentation", "knn_noise_proximity"]:
            if col in df.columns:
                row[f"ordering_rho[{col}]"] = round(spearmanr(ref.loc[common, col], df.loc[common, col]).statistic, 3)
        if ari:
            keys = [d for d in common if d[:11] in ari]
            if len(keys) > 5:
                y = [ari[d[:11]] for d in keys]
                row["r(D, meanARI)"] = round(pearsonr(df.loc[keys, "D"], y)[0], 3)
                row["rho(D, meanARI)"] = round(spearmanr(df.loc[keys, "D"], y).statistic, 3)
        out_rows.append(row)
    out = os.path.join(HERE, "tables"); os.makedirs(out, exist_ok=True)
    tab = pd.DataFrame(out_rows).set_index("perturbation")
    tab.to_csv(os.path.join(out, "q2_instrument_sensitivity.csv"))
    print("\nInstrument sensitivity (ordering rho vs default; r with mean ARI when available)\n")
    print(tab.to_string())

    # weights of the scalar: random +/-30% perturbations
    rng = np.random.RandomState(0)
    base_w = np.array([0.3, 0.3, 0.2, 0.2])
    comps = ref[[f"S_{c}" for c in COMP]].rename(columns=lambda s: s[2:])
    D0 = comps.apply(lambda r: rescore(r, base_w), axis=1)
    rhos, rs = [], []
    for _ in range(a.draws):
        w = base_w * rng.uniform(0.7, 1.3, 4)
        D = comps.apply(lambda r: rescore(r, w), axis=1)
        rhos.append(spearmanr(D0, D).statistic)
        if ari:
            keys = [d for d in ref.index if d[:11] in ari]
            rs.append(pearsonr(D.loc[keys], [ari[d[:11]] for d in keys])[0])
    wtab = pd.DataFrame({"ordering_rho_vs_default": rhos, **({"r_with_meanARI": rs} if rs else {})})
    wtab.describe(percentiles=[.025, .5, .975]).to_csv(os.path.join(out, "q2_weights_sensitivity.csv"))
    print(f"\nWeights +/-30% ({a.draws} draws): ordering rho min={min(rhos):.3f} median={np.median(rhos):.3f}"
          + (f"; r(D, meanARI) in [{min(rs):.3f}, {max(rs):.3f}]" if rs else ""))
    print(f"-> {out}/q2_instrument_sensitivity.csv, q2_weights_sensitivity.csv")


if __name__ == "__main__":
    main()
