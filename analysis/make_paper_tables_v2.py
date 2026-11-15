#!/usr/bin/env python
"""
Every table of the paper and its supplement, as LaTeX fragments, from the result CSVs.

    python analysis/make_paper_tables_v2.py [--results-root .] [--out analysis/tables_paper]

Writes one .tex file per table (tabular environment only; the caption lives in the paper),
so the manuscript and the supplement \\input them:
  tab_params.tex            every parameter and default (from the dataclasses)
  tab_ablation_F.tex        Table 3 (protocol F); tab_ablation_SDF.tex (supplement)
  tab_grouped_cv.tex        leave-one-dataset / configuration / regime R2
  tab_scalar_regimes.tex    scalar candidates across regimes
  tab_q2_instrument.tex     instrument sensitivity (needs analysis/tables/q2_instrument_sensitivity.csv)
  tab_divergence.tex        Table 5: r(D,ARI), r(D,Sil), mean ARI, degenerate, K-hat, F and [S]
  tab_failure_modes.tex     ok / silent / declared per regime, F
  tab_imbalance.tex         ARI and smallest-cluster F1 along rho_imb (F)
  tab_factor_descriptor.tex descriptor response to each intervention
  tab_thresholds.tex        critical levels with seed ranges (anisotropy F, overlap F)
  tab_crossgen.tex          cross-generator summary; tab_crossgen_full.tex
  tab_real.tex              19 real datasets
  tab_deep.tex              deep tier regime means and correlations
"""
from __future__ import annotations
import argparse, glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

FAMILY = {"KMeans": "centroid", "GMM": "centroid", "Spectral": "graph", "Ward": "connectivity", "Average": "connectivity",
          "Complete": "connectivity", "DBSCAN": "density", "HDBSCAN": "density", "OPTICS": "density", "DPC": "density"}
ORDER = ["KMeans", "GMM", "Spectral", "Ward", "Average", "Complete", "DBSCAN", "HDBSCAN", "OPTICS", "DPC"]
NAMES = {"KMeans": "K-Means"}
BASE = ["base_one_minus_silhouette", "base_knn_cv_error", "base_within_between_ratio"]
COMP = ["S_separation", "S_boundary", "S_shape", "S_noise"]
KNN = ["knn_edge_impurity", "knn_ambiguous_fraction", "knn_mean_impurity", "knn_fragmentation",
       "knn_density_heterogeneity", "knn_intra_density_spread", "knn_noise_fraction", "knn_noise_proximity"]
SETS = [("$1-$Silhouette (reference labels)", ["base_one_minus_silhouette"]), ("$k$-NN cross-validated error", ["base_knn_cv_error"]),
        ("within/between centroid ratio", ["base_within_between_ratio"]), ("all three baselines", BASE),
        ("$S_{\\text{sep}}$ alone", ["S_separation"]), ("$S_{\\text{bnd}}$ alone", ["S_boundary"]), ("$S_{\\text{shape}}$ alone", ["S_shape"]),
        ("$S_{\\text{noise}}$ alone", ["S_noise"]), ("$D$ (fixed-weight scalar)", ["D_dataset"]), ("four components, separately", COMP),
        ("$k$-NN descriptors", KNN), ("components $+$ $k$-NN descriptors", COMP + KNN), ("full profile ($+$ baselines)", COMP + KNN + BASE)]


def load(root, rel):
    f = sorted(glob.glob(os.path.join(root, rel))); return pd.read_csv(f[0], float_precision="round_trip") if f else None   # exact parsing: same doubles under every pandas version


def f2(v, d=2):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def sgn(v, d=2):
    return "--" if np.isnan(v) else f"${v:+.{d}f}$"


def write(out, name, header, rows, colspec):
    body = "\n".join(rows)
    txt = f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    open(os.path.join(out, name), "w").write(txt); print(f"  {name}")


def loo_r2(X, y, cv="loo", groups=None):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict, KFold
    X = np.asarray(X, float); y = np.asarray(y, float); ok = ~np.isnan(X).any(1) & ~np.isnan(y); X, y = X[ok], y[ok]
    if len(y) < 8: return np.nan
    m = RidgeCV(alphas=np.logspace(-3, 2, 12))
    if cv == "loo": p = cross_val_predict(m, X, y, cv=LeaveOneOut())
    elif cv == "k10": p = cross_val_predict(m, X, y, cv=KFold(10, shuffle=True, random_state=0))
    else: p = cross_val_predict(m, X, y, cv=LeaveOneGroupOut(), groups=np.asarray(groups)[ok])
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ds_level(df, extra=()):
    """Dataset-level means over the paper's suite pool (DPC is reported separately)."""
    df = df[df.algorithm != "DPC"] if "algorithm" in df else df
    key = ["config_id", "seed"] if "config_id" in df else ["value", "seed"]
    cols = [c for c in df.columns if c.startswith(("S_", "knn_", "base_", "D_"))]
    agg = df.groupby(key).agg({"ARI": "mean", **{c: "first" for c in cols}}).reset_index()
    # CSVs written before v2.1.7 hold NaN for noise_proximity on noise-free datasets; the profile now returns 0
    return agg.fillna({"knn_noise_proximity": 0.0})


def pooled_frames(R):
    """Dataset-level frames (mean ARI over the classical pool, DPC excluded as in the paper's pool
    means, plus the profile) for the suite (F), the sweeps (F), the noise grid (S) and their union."""
    C = "benchmarks/classical/results"
    E1F = load(R, f"{C}/exp1_difficulty_validation/exp1_results_fixed.csv")
    sw = [load(R, f"{C}/sweeps_v2/sweep_{k}_fixed.csv") for k in ["overlap", "imbalance", "inter_density", "intra_density"]]
    ANIF = load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_fixed_anisotropy.csv")
    NOISE = load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_silhouette_noise.csv")
    if E1F is None or any(x is None for x in sw) or ANIF is None or NOISE is None:
        return None
    def agg_of(df, key):
        df = df[df.algorithm != "DPC"]
        cols = [c for c in df.columns if c.startswith(("S_", "knn_", "base_", "D_"))]
        out = df.groupby(key).agg({"ARI": "mean", **{c: "first" for c in cols}})
        return out.fillna({"knn_noise_proximity": 0.0})
    suite = agg_of(E1F, ["config_id", "seed"]).reset_index(drop=True)
    sweeps = pd.concat([agg_of(d, ["value", "seed"]).reset_index(drop=True) for d in sw + [ANIF]], ignore_index=True)
    noise = agg_of(NOISE, ["value_eta", "value_eta_int", "seed"]).reset_index(drop=True)
    pooled = pd.concat([suite, sweeps, noise], ignore_index=True)
    return {"suite": suite, "sweeps": sweeps, "noise": noise, "pooled": pooled}


def scalar_candidates(d):
    phi = lambda x, m: np.clip(x / m, 0, 1)
    dk = (phi(d.knn_edge_impurity, .5) + d.S_boundary + d.S_noise) / 3
    return {"D": d.D_dataset, "D_knn": dk, "one_minus_silhouette": d.base_one_minus_silhouette, "knn_cv_error": d.base_knn_cv_error}


def pooled_summary(R):
    """Numbers quoted in Section 6.1 for the pooled set (read by audit_paper_claims_v2.py)."""
    fr = pooled_frames(R)
    if fr is None: return None
    P = fr["pooled"]; out = {"N_pooled": len(P), "N_sweeps": len(fr["sweeps"])}
    for k, x in scalar_candidates(P).items():
        ok = x.notna(); out[f"r_pooled_{k}"] = float(pearsonr(x[ok], P.ARI[ok])[0])
    for k, x in scalar_candidates(fr["noise"]).items():
        ok = x.notna() & (x.std() > 0)
        out[f"r_noise_{k}"] = float(pearsonr(x[ok], fr["noise"].ARI[ok])[0]) if ok.sum() > 5 and x[ok].std() > 0 else float("nan")
    out["R2_k10_components_knn"] = float(loo_r2(P[COMP + KNN], P.ARI, "k10"))
    out["R2_k10_baselines"] = float(loo_r2(P[BASE], P.ARI, "k10"))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-root", default="."); ap.add_argument("--out", default="analysis/tables_paper")
    a = ap.parse_args(); R, out = a.results_root, a.out; os.makedirs(out, exist_ok=True)
    C = "benchmarks/classical/results"
    E1 = {p: load(R, f"{C}/exp1_difficulty_validation/exp1_results{s}.csv") for p, s in [("S", ""), ("D", "_dbcv"), ("F", "_fixed")]}
    DPC1 = {p: load(R, f"{C}/exp1_difficulty_validation/exp1_results{s}_DPC.csv") for p, s in [("S", ""), ("F", "_fixed")]}
    for p in ("S", "F"):   # merge DPC rows when run separately
        if E1[p] is not None and DPC1[p] is not None and "DPC" not in E1[p].algorithm.unique():
            E1[p] = pd.concat([E1[p], DPC1[p]], ignore_index=True)
    print("tables ->", out)

    # ---------------- params
    try:
        sys.path.insert(0, R); from dataclasses import fields
        from clustaudit.config.params import StructuralParams, NoiseParams, AmbiguityParams, DensityParams
        rows = []
        for cls, name in [(StructuralParams, "$\\Theta_{\\text{struct}}$"), (DensityParams, "$\\Theta_{\\text{struct}}$ (density)"), (NoiseParams, "$\\Theta_{\\text{noise}}$"), (AmbiguityParams, "$\\Theta_{\\text{amb}}$")]:
            inst = cls()
            for f in fields(cls):
                if f.name in ("cluster_params", "manifold_params", "density_params", "enable_adaptive_density", "noise_threshold", "internal_noise_scale", "internal_noise_classification_threshold"): continue
                v = getattr(inst, f.name); rows.append(f"{name} & \\texttt{{{f.name.replace('_', '_')}}} & {str(v).replace('_', '_')} \\\\".replace("_", "\\_").replace("\\\\_", "\\_"))
        write(out, "tab_params.tex", "Set & Parameter & Default", rows, "@{}lll@{}")
    except Exception as e:
        print("  tab_params skipped:", e)

    # ---------------- ablation F, SDF, grouped CV
    if all(E1[p] is not None for p in "SDF"):
        pool9 = [a for a in ORDER if a != "DPC"]   # the suite pool of the paper (DPC is reported separately)
        tabs = {}
        for p in "SDF":
            df = E1[p][E1[p].algorithm.isin(pool9)].copy(); df["family"] = df.algorithm.map(FAMILY)
            groups = {"All": df} | {f: g for f, g in df.groupby("family")}
            tabs[p] = {}
            for gname, g in groups.items():
                agg = ds_level(g)
                tabs[p][gname] = {s: loo_r2(agg[[c for c in cols if c in agg]], agg.ARI) for s, cols in SETS}
        fams = ["All", "centroid", "connectivity", "density", "graph"]
        best = {g: max(tabs["F"][g][s] for s, _ in SETS) for g in fams}
        bold = lambda v, g: f"\\textbf{{{v:.2f}}}" if abs(v - best[g]) < 5e-3 else f2(v)
        rows = [f"{s} & " + " & ".join(bold(tabs['F'][g][s], g) for g in fams) + " \\\\" for s, _ in SETS]
        for i in (3, 9): rows.insert(i + 1, "\\midrule")
        write(out, "tab_ablation_F.tex", "Predictor set & All & Centroid & Connectivity & Density & Graph", rows, "@{}lccccc@{}")
        rows = [f"{s} & " + " & ".join(f2(tabs[p][g][s]) for p in "SDF" for g in fams) + " \\\\" for s, _ in SETS]
        head = "Predictor set & " + " & ".join(f"{p}:{g[:4]}" for p in "SDF" for g in fams)
        write(out, "tab_ablation_SDF.tex", head, rows, "@{}l" + "c" * 15 + "@{}")
        agg = ds_level(E1["F"][E1["F"].algorithm.isin(pool9)]); reg = agg.config_id // 5
        rows = []
        for s, cols in [("$1-$Silhouette (reference)", ["base_one_minus_silhouette"]), ("$D$", ["D_dataset"]), ("four components", COMP), ("$k$-NN descriptors", KNN), ("components $+$ $k$-NN", COMP + KNN), ("full profile ($+$ baselines)", COMP + KNN + BASE)]:
            X = agg[cols]; rows.append(f"{s} & {f2(loo_r2(X, agg.ARI))} & {f2(loo_r2(X, agg.ARI, 'g', agg.config_id))} & {f2(loo_r2(X, agg.ARI, 'g', reg))} \\\\")
        write(out, "tab_grouped_cv.tex", "Predictor set & leave-one-dataset-out & leave-one-configuration-out & leave-one-regime-out", rows, "@{}lccc@{}")

    # ---------------- scalar candidates across regimes
    sw = [load(R, f"{C}/sweeps_v2/sweep_{k}_fixed.csv") for k in ["overlap", "imbalance", "inter_density", "intra_density"]]
    ANIF = load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_fixed_anisotropy.csv"); NOISE = load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_silhouette_noise.csv")
    fr = pooled_frames(R)
    if fr is not None:
        names = {"D": "$D$", "D_knn": "$D_{\\text{knn}}$ (no Silhouette)", "one_minus_silhouette": "$1-$Silhouette (reference)", "knn_cv_error": "$k$-NN CV error"}
        rows = []
        for key, label in names.items():
            cells = []
            for part in ("suite", "sweeps", "noise", "pooled"):
                d = fr[part]; x = scalar_candidates(d)[key]; ok = x.notna()
                cells.append(sgn(pearsonr(x[ok], d.ARI[ok])[0]) if ok.sum() > 5 and x[ok].std() > 0 else "--")
            rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        ps = pooled_summary(R)
        rows.append("\\midrule")
        rows.append(f"ten-fold $R^2$: components $+$ $k$-NN & & & & {ps['R2_k10_components_knn']:.2f} \\\\")
        rows.append(f"ten-fold $R^2$: three baselines & & & & {ps['R2_k10_baselines']:.2f} \\\\")
        n = {k: len(v) for k, v in fr.items()}
        write(out, "tab_scalar_regimes.tex", f"Scalar & suite (F), $N={n['suite']}$ & sweeps (F), $N={n['sweeps']}$ & noise grid (S), $N={n['noise']}$ & pooled, $N={n['pooled']}$", rows, "@{}lcccc@{}")

    # ---------------- instrument sensitivity
    Q2 = load(R, "analysis/tables/q2_instrument_sensitivity.csv")
    if Q2 is not None:
        q = Q2.set_index(Q2.columns[0])
        cols = [c for c in q.columns if c.startswith("ordering_rho")] + ["r(D, meanARI)"]
        head = "Perturbation & " + " & ".join(c.replace("ordering_rho[", "$\\rho$[").replace("knn_", "").replace("_", "\\_").replace("r(D, meanARI)", "$r(D,\\text{ARI})$") for c in cols)
        PERT = {"default": "default instrument", "knn k=5": "$k$-NN, $k = 5$", "knn k=2sqrt(n)": "$k$-NN, $k = 2\\sqrt{n}$", "random forest": "random forest", "svm": "SVM",
                "tau_core=0.90": "$\\tau_{\\text{core}} = 0.90$", "tau_core=0.99": "$\\tau_{\\text{core}} = 0.99$", "q_gap=0.05": "$q_{\\text{gap}} = 0.05$", "q_gap=0.25": "$q_{\\text{gap}} = 0.25$",
                "in-sample confidence": "in-sample confidence", "folds=10": "10 folds"}
        rows = [f"{PERT.get(i, str(i).replace('_', chr(92) + '_'))} & " + " & ".join(f2(q.loc[i, c], 3) for c in cols) + " \\\\" for i in q.index]
        write(out, "tab_q2_instrument.tex", head, rows, "@{}l" + "c" * len(cols) + "@{}")

    # ---------------- divergence table (F with S in brackets)
    if E1["F"] is not None and E1["S"] is not None:
        rows = []
        for alg in ORDER:
            cells = [NAMES.get(alg, alg)]
            for col in ("ARI", "Silhouette"):
                vals = []
                for p in "FS":
                    d = E1[p][E1[p].algorithm == alg].dropna(subset=[col]); vals.append(pearsonr(d.D_dataset, d[col])[0] if len(d) > 5 else np.nan)
                cells.append(f"{sgn(vals[0])} [{sgn(vals[1])}]")
            for col in ("ARI", "degenerate", "n_clusters_found"):
                vals = [E1[p][E1[p].algorithm == alg][col].mean() for p in "FS"]
                cells.append(f"{f2(vals[0], 1 if col == 'n_clusters_found' else 2)} [{f2(vals[1], 1 if col == 'n_clusters_found' else 2)}]")
            if not E1["F"][E1["F"].algorithm == alg].empty: rows.append(" & ".join(cells) + " \\\\")
        write(out, "tab_divergence.tex", "Algorithm & $r(D,\\text{ARI})$ & $r(D,\\text{Sil})$ & mean ARI & degenerate & $\\hat K$", rows, "@{}lccccc@{}")

    # ---------------- failure modes (F)
    if E1["F"] is not None and "failure_mode" in E1["F"]:
        rows = []
        for alg in ORDER:
            d = E1["F"][E1["F"].algorithm == alg]
            if d.empty: continue
            cells = [NAMES.get(alg, alg)]
            for mode in ("ok", "silent", "declared"):
                for reg in ("easy", "medium", "hard"):
                    dd = d[d.difficulty == reg]; cells.append(f2((dd.failure_mode == mode).mean()))
            cells.append(f2(d[d.difficulty == "hard"].min_cluster_f1.mean()))
            rows.append(" & ".join(cells) + " \\\\")
        head = (" & \\multicolumn{3}{c}{ok} & \\multicolumn{3}{c}{silent} & \\multicolumn{3}{c}{detectable} & F1$_{\\min}$ hard \\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\nAlgorithm & E & M & H & E & M & H & E & M & H &")
        write(out, "tab_failure_modes.tex", head, rows, "@{}l ccc ccc ccc c@{}")

    # ---------------- imbalance (F)
    IMB = load(R, f"{C}/sweeps_v2/sweep_imbalance_fixed.csv"); IMBD = load(R, f"{C}/sweeps_v2/sweep_imbalance_fixed_DPC.csv")
    if IMB is not None:
        if IMBD is not None and "DPC" not in IMB.algorithm.unique(): IMB = pd.concat([IMB, IMBD])
        vals = sorted(IMB.value.unique()); rows = []
        for alg in ORDER:
            d = IMB[IMB.algorithm == alg]
            if d.empty: continue
            rows.append(NAMES.get(alg, alg) + " & " + " & ".join(f2(d[d.value == v].ARI.mean()) for v in vals) + " & & " + " & ".join(f2(d[d.value == v].min_cluster_f1.mean()) for v in vals) + " \\\\")
        head = "$\\rho_{\\text{imb}}$ & " + " & ".join(f"{v:g}" for v in vals) + " & & " + " & ".join(f"{v:g}" for v in vals)
        head = (" & \\multicolumn{%d}{c}{inlier ARI} & & \\multicolumn{%d}{c}{F1 of the smallest cluster} \\\\\n\\cmidrule{2-%d}\\cmidrule{%d-%d}\n" % (len(vals), len(vals), len(vals) + 1, len(vals) + 3, 2 * len(vals) + 2)) + head
        write(out, "tab_imbalance.tex", head, rows, "@{}l" + "c" * len(vals) + "c" + "c" * len(vals) + "@{}")
        if "NMI" in IMB and "min_cluster_recall" in IMB:
            rows = []
            for alg in ORDER:
                d = IMB[IMB.algorithm == alg]
                if d.empty: continue
                rows.append(NAMES.get(alg, alg) + " & " + " & ".join(f2(d[d.value == v].NMI.mean()) for v in vals) + " & & " + " & ".join(f2(d[d.value == v].min_cluster_recall.mean()) for v in vals) + " \\\\")
            head2 = head.replace("inlier ARI", "NMI").replace("F1 of the smallest cluster", "recall of the smallest cluster")
            write(out, "tab_imbalance_nmi_recall.tex", head2, rows, "@{}l" + "c" * len(vals) + "c" + "c" * len(vals) + "@{}")
        if "min_cluster_f1_inlier" in IMB and IMB.min_cluster_f1_inlier.notna().any():
            rows = []
            for alg in ORDER:
                d = IMB[IMB.algorithm == alg]
                if d.empty: continue
                rows.append(NAMES.get(alg, alg) + " & " + " & ".join(f2(d[d.value == v].min_cluster_f1.mean()) for v in vals) + " & & " + " & ".join(f2(d[d.value == v].min_cluster_f1_inlier.mean()) for v in vals) + " \\\\")
            write(out, "tab_imbalance_f1_conventions.tex", head.replace("inlier ARI", "F1, all points").replace("F1 of the smallest cluster", "F1, inliers only"), rows, "@{}l" + "c" * len(vals) + "c" + "c" * len(vals) + "@{}")
        IMB3 = load(R, f"{C}/sweeps_v2/sweep_imbalance_fixed_n3000.csv")
        if IMB3 is not None and "min_cluster_f1" in IMB3:
            vals3 = sorted(IMB3.value.unique()); rows = []
            for alg in ORDER:
                d = IMB3[IMB3.algorithm == alg]
                if d.empty: continue
                rows.append(NAMES.get(alg, alg) + " & " + " & ".join(f2(d[d.value == v].ARI.mean()) for v in vals3) + " & & " + " & ".join(f2(d[d.value == v].min_cluster_f1.mean()) for v in vals3) + " \\\\")
            write(out, "tab_imbalance_n3000.tex", head, rows, "@{}l" + "c" * len(vals3) + "c" + "c" * len(vals3) + "@{}")

    # ---------------- factor x descriptor
    desc = ["S_separation", "S_boundary", "S_shape", "S_noise", "knn_edge_impurity", "knn_ambiguous_fraction", "knn_fragmentation", "knn_density_heterogeneity", "knn_intra_density_spread", "knn_noise_proximity", "knn_intrinsic_dim_twonn", "base_one_minus_silhouette", "base_knn_cv_error"]
    E2 = load(R, f"{C}/exp2_sensitivity_analysis/exp2_results.csv"); DIMF = load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_silhouette_fixedn.csv")
    items = []
    for name, df, key in [("overlap $\\sigma$", sw[0], "value"), ("imbalance $\\rho_{\\text{imb}}$", sw[1], "value"), ("inter-density $\\delta_{\\text{inter}}$", sw[2], "value"), ("intra-density $s_{\\text{intra}}$", sw[3], "value"),
                          ("anisotropy $\\alpha$", E2[E2.parameter == "anisotropy"] if E2 is not None else None, "value"), ("dimension $d$ ($n$ fixed)", DIMF, "value"),
                          ("noise amount $\\eta$", NOISE, "value_eta"), ("noise placement $\\eta_{\\text{int}}$ ($\\eta = 0.3$)", NOISE[NOISE.value_eta == 0.3] if NOISE is not None else None, "value_eta_int")]:
        if df is None: continue
        d = df.groupby([key, "seed"])[desc].first().reset_index()
        cells = []
        for c in desc:
            if d[c].isna().all() or d[c].nunique(dropna=True) <= 1: cells.append("--"); continue
            r = spearmanr(d[key], d[c], nan_policy="omit").statistic
            cells.append("--" if not np.isfinite(r) else (f"\\textbf{{{r:+.2f}}}" if abs(r) > .5 else f"{r:+.2f}"))
        items.append(name + " & " + " & ".join(cells) + " \\\\")
    if items:
        write(out, "tab_factor_descriptor.tex", "Sweep & $S_{\\text{sep}}$ & $S_{\\text{bnd}}$ & $S_{\\text{shape}}$ & $S_{\\text{noise}}$ & impurity & ambig. & fragm. & dens.\\,het. & intra\\,dens. & noise\\,prox. & intr.\\,dim & 1$-$Sil & kNN err", items, "@{}l" + "r" * 13 + "@{}")

    # ---------------- critical thresholds with seed ranges
    def crit(df, thr=.5):
        res = {}
        for alg, g in df.groupby("algorithm"):
            per = []
            for s, gs in g.groupby("seed"):
                m = gs.groupby("value").ARI.mean(); b = m[m < thr]; per.append(b.index.min() if len(b) else np.inf)
            m = g.groupby("value").ARI.mean(); b = m[m < thr]
            res[alg] = (b.index.min() if len(b) else np.inf, min(per), max(per))
        return res
    if ANIF is not None and sw[0] is not None:
        ca, co = crit(ANIF), crit(sw[0]); fmt = lambda v: "$>$max" if np.isinf(v) else f"{v:g}"
        rows = [f"{NAMES.get(a, a)} & {fmt(ca[a][0])} [{fmt(ca[a][1])}, {fmt(ca[a][2])}] & {fmt(co[a][0])} [{fmt(co[a][1])}, {fmt(co[a][2])}] \\\\" for a in ORDER if a in ca and a in co]
        write(out, "tab_thresholds.tex", "Algorithm & $\\alpha_{\\text{crit}}$ (F) [seed range] & $\\sigma_{\\text{crit}}$ (F) [seed range]", rows, "@{}lcc@{}")

    # ---------------- audit cards: the standardized output of the battery, one row per method (protocol F)
    def with_dpc(df, name):
        d = load(R, f"{C}/sweeps_v2/{name}_DPC.csv")
        return pd.concat([df, d]) if (df is not None and d is not None and "DPC" not in df.algorithm.unique()) else df
    OV = with_dpc(sw[0], "sweep_overlap_fixed"); ID = with_dpc(sw[2], "sweep_inter_density_fixed"); IM = with_dpc(sw[1], "sweep_imbalance_fixed")
    ETA = load(R, f"{C}/sweeps_v2/sweep_eta_int_fixed_eta0.3.csv")
    if all(x is not None for x in (OV, ID, IM, ETA, ANIF)):
        fmt = lambda v: "$>$max" if np.isinf(v) else f"{v:g}"
        cards = {k: crit(df) for k, df in [("ov", OV), ("an", ANIF), ("id", ID)]}
        rows = []
        for alg in ORDER:
            if alg not in cards["ov"]: continue
            cells = [NAMES.get(alg, alg)]
            for k, df in (("ov", OV), ("an", ANIF), ("id", ID)):
                c = cards[k].get(alg); first = df.value.min()
                if c is None: cells.append("n/a")
                elif c[0] == first and c[2] == first: cells.append(f"$\\le {first:g}$")   # below 0.5 from the first level in every seed
                else: cells.append(f"{fmt(c[0])} [{fmt(c[1])}, {fmt(c[2])}]")
            e = ETA[ETA.algorithm == alg]; cells.append(f"{f2(e[e.value == 0].ARI.mean())} / {f2(e[e.value == 1].ARI.mean())}" if not e.empty else "n/a")
            im = IM[IM.algorithm == alg]; vmax = im.value.max(); cells.append(f"{f2(im[im.value == vmax].ARI.mean())} / {f2(im[im.value == vmax].min_cluster_f1.mean())}")
            o = OV[(OV.algorithm == alg) & (OV.value == OV.value.max())].copy()
            if "failure_mode" not in o or o.failure_mode.isna().all():   # CSVs written before the column existed: same rule as evaluate_partition
                o["failure_mode"] = np.where(o.degenerate.astype(bool), "declared", np.where(o.ARI < 0.3, "silent", "ok"))
            mode = o.failure_mode.value_counts(normalize=True).to_dict()
            lab = max(mode, key=mode.get) if len(mode) else "--"
            cells.append({"declared": "detectable", "silent": "silent", "ok": "ok"}.get(lab, lab) + (f" ({mode[lab]:.2f})" if len(mode) else ""))
            rows.append(" & ".join(cells) + " \\\\")
        head = ("Method & $\\sigma_{\\text{crit}}$ & $\\alpha_{\\text{crit}}$ & $\\delta_{\\text{crit}}$ & noise placement & imbalance $\\rho = 5$ & failure at $\\sigma_{\\max}$ \\\\\n"
                " & [seed range] & [seed range] & [seed range] & ARI dist.\\,/\\,bridge & ARI / F1$_{\\min}$ & mode (fraction)")
        write(out, "tab_audit_cards.tex", head, rows, "@{}lcccccc@{}")

    # ---------------- calibrated generation (exp6)
    E6 = load(R, f"{C}/exp6_calibrated_generation/datasets.csv"); P6 = load(R, f"{C}/exp6_calibrated_generation/pool.csv")
    if E6 is not None:
        E6["err_imp"] = (E6.knn_edge_impurity - E6.target_edge_impurity).abs(); E6["err_sn"] = (E6.S_noise - E6.target_S_noise).abs()
        rows = []
        for route in ("parametric", "calibrated"):
            d = E6[E6.route == route]; sd = d.groupby("target_id")[["knn_edge_impurity", "S_noise"]].std().mean()
            cells = [route, f"{int(d.hit.sum())}/{len(d)} ({d.hit.mean():.3f})", f2(d.err_imp.mean(), 3), f2(d.err_sn.mean(), 3), f2(sd.knn_edge_impurity, 3), f2(sd.S_noise, 3), f"{d.n_evals.mean():.1f}"]
            if P6 is not None:
                p = P6[P6.route == route]; sdA = p.groupby(["target_id", "algorithm"]).ARI.std().mean(); cells.append(f2(sdA, 3))
            rows.append(" & ".join(cells) + " \\\\")
        head = "Route & hits & MAE impurity & MAE $S_{\\text{noise}}$ & spread impurity & spread $S_{\\text{noise}}$ & evals / dataset" + (" & spread of ARI" if P6 is not None else "")
        write(out, "tab_calibrated.tex", head, rows, "@{}l" + "c" * (len(rows[0].split("&")) - 1) + "@{}")
        rows = []
        for ti, g in E6.groupby("target_id"):
            r = g.iloc[0]; par = g[g.route == "parametric"]; cal = g[g.route == "calibrated"]
            rows.append(f"{r.target_edge_impurity:g} / {r.target_S_noise:g} & {f2(par.hit.mean())} & {f2(par.err_imp.mean(), 3)} & {f2(cal.hit.mean())} & {f2(cal.err_imp.mean(), 3)} & {cal.n_evals.mean():.1f} & "
                        + f"$\\sigma$ {cal.param_sigma.mean():.2f} [{cal.param_sigma.min():.2f}, {cal.param_sigma.max():.2f}], $\\alpha$ {cal.param_alpha.mean():.1f}, $\\eta$ {cal.param_eta.mean():.2f}, $\\eta_{{\\text{{int}}}}$ {cal.param_eta_int.mean():.2f} \\\\")
        write(out, "tab_calibrated_targets.tex", "Target impurity / $S_{\\text{noise}}$ & \\multicolumn{2}{c}{parametric: hit, MAE} & \\multicolumn{3}{c}{calibrated: hit, MAE, evals} & parameters found (mean over seeds)", rows, "@{}lcccccl@{}")

    # ---------------- cross-generator
    XG = load(R, "benchmarks/cross-generator/results/crossgen_results.csv")
    if XG is not None:
        XG["family"] = XG.algorithm.map(FAMILY)
        def corr(d, col):
            agg = d.groupby(["native_param", "seed"]).agg(ARI=("ARI", "mean"), v=(col, "first")); return pearsonr(agg.v, agg.ARI)[0]
        preds = [("native parameter", "native_param"), ("reference Silhouette", "silhouette_ref"), ("$D$", "D_dataset"), ("$D_{\\text{knn}}$", "D_knn")]
        rows = []
        for pname, col in preds:
            cells = [sgn(corr(XG[(XG.generator == g) & (XG.protocol == p)], col)) for g in ("mdcgen", "hawks") for p in ("fixed", "silhouette")]
            fam = [f"{sgn(corr(XG[(XG.generator == 'mdcgen') & (XG.protocol == 'fixed') & (XG.family == f)], col))} / {sgn(corr(XG[(XG.generator == 'hawks') & (XG.protocol == 'fixed') & (XG.family == f)], col))}" for f in ("centroid", "connectivity", "density", "graph")]
            rows.append(pname + " & " + " & ".join(cells + fam) + " \\\\")
        write(out, "tab_crossgen.tex", "Predictor & MDCGen (F) & MDCGen (S) & HAWKS (F) & HAWKS (S) & centroid & connect. & density & graph", rows, "@{}lcccccccc@{}")
        rows = []
        for (g, p), d in XG.groupby(["generator", "protocol"]):
            for grp, x in ({"all": d} | {f: y for f, y in d.groupby("family")}).items():
                rows.append(f"{g} & {p} & {grp} & " + " & ".join(sgn(corr(x, col)) for _, col in preds) + " \\\\")
        write(out, "tab_crossgen_full.tex", "Generator & Protocol & Group & native & Silhouette & $D$ & $D_{\\text{knn}}$", rows, "@{}lllcccc@{}")

    # ---------------- real data
    REAL = load(R, "benchmarks/real-data/results/transfer_results.csv"); REALPA = load(R, "benchmarks/real-data/results/transfer_per_algo.csv")
    RELAB = load(R, "benchmarks/real-data/results/transfer_results_relabel.csv")
    if REAL is not None and REALPA is not None:
        df = (REAL[REAL.relabel == False] if "relabel" in REAL else REAL).set_index("dataset"); deg = REALPA.groupby("dataset").degenerate.mean()
        rows = []
        for ds, r in df.iterrows():
            extra = f2(RELAB.set_index('dataset').loc[ds, 'D_dataset']) if RELAB is not None and ds in RELAB.dataset.values else "--"
            dst = ds.replace("_", "\\_")   # (no backslash inside f-string expressions: Python < 3.12)
            rows.append(f"\\texttt{{{dst}}} & {int(r.n)} & {int(r.d)} & {int(r.K)} & {f2(r.D_dataset)} & {f2(r.knn_edge_impurity)} & {f2(r.knn_intrinsic_dim_twonn, 1)} & {f2(r.clean_median)} & {f2(deg.get(ds, np.nan))} & {extra} \\\\")
        write(out, "tab_real.tex", "Dataset & $n$ & $d$ & $K$ & $D$ (raw) & edge imp. & intr.\\,dim & median Sil & degen.\\,frac & $D$ (smoothed)", rows, "@{}lrrrcccccc@{}")

    # ---------------- deep
    DEEP = load(R, "benchmarks/deep/results/deep_results.csv")
    if DEEP is not None and E1["F"] is not None:
        seeds = [42, 123, 456, 789, 1011]; e = E1["F"].copy(); e["idx"] = e.config_id * 5 + e.seed.map({s: i for i, s in enumerate(seeds)})
        prof = e.groupby("idx").D_dataset.first(); D = DEEP.copy(); D["idx"] = D.dataset.str.extract(r"dataset_(\d+)_").astype(int)
        dm = D.groupby(["idx", "method"]).agg(ARI=("ARI", "mean"), Sil=("Silhouette", "mean"), K=("n_clusters_found", "mean")).reset_index(); dm["D"] = dm.idx.map(prof); dm["reg"] = dm.idx // 25
        rows = []
        for m in ("IDEC", "DCN", "DipDECK"):
            g = dm[dm.method == m]
            rows.append(f"{m} & {sgn(pearsonr(g.D, g.ARI)[0])} & {sgn(spearmanr(g.D, g.ARI).statistic)} & {sgn(pearsonr(g.dropna(subset=['Sil']).D, g.dropna(subset=['Sil']).Sil)[0])} & "
                        + " & ".join(f2(g[g.reg == r].ARI.mean()) for r in range(3)) + " & " + " & ".join(f2(g[g.reg == r].K.mean(), 1) for r in range(3)) + " \\\\")
        write(out, "tab_deep.tex", "Method & $r(D,\\text{ARI})$ & $\\rho(D,\\text{ARI})$ & $r(D,\\text{Sil})$ & ARI E & ARI M & ARI H & $\\hat K$ E & $\\hat K$ M & $\\hat K$ H", rows, "@{}lccccccccc@{}")


if __name__ == "__main__":
    main()
