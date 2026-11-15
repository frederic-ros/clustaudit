#!/usr/bin/env python
"""Q4 analysis: native parameter vs profile as predictors of ARI on MDCGen / HAWKS data.

    python benchmarks/cross-generator/analyze_crossgen.py  [--protocol fixed]
-> results/crossgen_table.csv and a per-family table printed to the console.
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILY = {"KMeans": "centroid", "GMM": "centroid", "Spectral": "graph", "Ward": "connectivity",
          "Average": "connectivity", "DBSCAN": "density", "HDBSCAN": "density", "OPTICS": "density", "DPC": "density"}
PREDS = {"native parameter": "native_param", "reference Silhouette": "silhouette_ref", "D_dataset": "D_dataset",
         "D_knn (no Silhouette)": "D_knn", "edge impurity": "knn_edge_impurity", "S_bnd": "S_boundary"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--protocol", default="fixed"); a = ap.parse_args()
    df = pd.read_csv(os.path.join(HERE, "results", "crossgen_results.csv"))
    df = df[df.protocol == a.protocol].copy(); df["family"] = df.algorithm.map(FAMILY)
    rows = []
    for g, dg in df.groupby("generator"):
        groups = {"all": dg} | {f: x for f, x in dg.groupby("family")}
        for gname, x in groups.items():
            agg = x.groupby(["native_param", "seed"]).agg({"ARI": "mean", **{v: "first" for v in PREDS.values()}})
            for pname, col in PREDS.items():
                r = pearsonr(agg[col], agg.ARI)[0]; rho = spearmanr(agg[col], agg.ARI).statistic
                rows.append({"generator": g, "group": gname, "predictor": pname, "r": round(r, 3), "rho": round(rho, 3), "N": len(agg)})
    t = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", f"crossgen_table_{a.protocol}.csv"); t.to_csv(out, index=False)
    for g in t.generator.unique():
        print(f"\n== {g} (protocol {a.protocol}) : |r| with mean ARI ==")
        print(t[t.generator == g].pivot(index="predictor", columns="group", values="r").reindex(list(PREDS)).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
