#!/usr/bin/env python
"""
Every figure of the paper and its supplement, from the result CSVs.

    python analysis/make_paper_figures_v2.py [--results-root .] [--out analysis/figures_paper]

Figures (main text):  factors_2d, noise_crossover, noise_grid, overlap_SF, anisotropy_SF,
                      dimension_two_regimes, density_imbalance_F, transfer_real, deep
Figures (supplement): noise_grid_conventions, dimension_silhouette_K, imbalance_n3000,
                      overlap_profile
Style: one colour per algorithmic family (centroid = blues, connectivity = greens,
density = reds/oranges, graph/peaks = purples), one marker per method, 11-pt fonts,
seed spread as a light band rather than error bars.
"""
from __future__ import annotations
import argparse, glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {  # colour, marker, linestyle
    "KMeans":   ("#1f5fa8", "o", "-"),  "GMM":     ("#6baed6", "s", "-"),
    "Ward":     ("#2a924a", "^", "-"),  "Average": ("#8fd18b", "v", "--"), "Complete": ("#c7e9c0", "<", ":"),
    "DBSCAN":   ("#c0392b", "D", "-"),  "HDBSCAN": ("#e8743b", "p", "-"),  "OPTICS":   ("#f3b562", "h", "--"),
    "Spectral": ("#6a3d9a", "*", "-"),  "DPC":     ("#b57edc", "X", "--"),
    "IDEC": ("#1f5fa8", "o", "-"), "DCN": ("#2a924a", "s", "-"), "DipDECK": ("#c0392b", "D", "-"),
}
ORDER = ["KMeans", "GMM", "Ward", "Average", "Spectral", "DPC", "DBSCAN", "HDBSCAN", "OPTICS"]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.grid": True, "grid.alpha": .25,
                     "lines.linewidth": 1.8, "lines.markersize": 6, "figure.dpi": 110})


def line(ax, df, x, alg, label=None, band=True, y="ARI"):
    d = df[df.algorithm == alg]
    if d.empty: return
    g = d.groupby(x)[y].agg(["mean", "std"])
    c, m, ls = STYLE.get(alg, ("k", "o", "-"))
    ax.plot(g.index, g["mean"], color=c, marker=m, ls=ls, label=label or alg, zorder=3)
    if band:
        ax.fill_between(g.index, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=c, alpha=.07, lw=0)


def legend_families(ax, algs, **kw):
    h, l = ax.get_legend_handles_labels()
    order = [a for a in ORDER if a in l]
    ax.legend([h[l.index(a)] for a in order], order, ncol=kw.pop("ncol", 3), frameon=False, **kw)


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig); print(f"  {name}")


def load(root, rel):
    f = sorted(glob.glob(os.path.join(root, rel)))
    return pd.read_csv(f[0], float_precision="round_trip") if f else None   # exact parsing: same doubles under every pandas version


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-root", default="."); ap.add_argument("--out", default="analysis/figures_paper")
    a = ap.parse_args(); R = a.results_root; out = a.out; os.makedirs(out, exist_ok=True)
    C = "benchmarks/classical/results"
    NOISE = load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_silhouette_noise.csv")
    EIF = load(R, f"{C}/sweeps_v2/sweep_eta_int_fixed_eta0.3.csv")
    OVS, OVF = load(R, f"{C}/sweeps_v2/sweep_overlap_silhouette.csv"), load(R, f"{C}/sweeps_v2/sweep_overlap_fixed.csv")
    E2 = load(R, f"{C}/exp2_sensitivity_analysis/exp2_results.csv")
    DIMF = load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_silhouette_fixedn.csv")
    ANIF = load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_fixed_anisotropy.csv")
    IMB, IMB3 = load(R, f"{C}/sweeps_v2/sweep_imbalance_fixed.csv"), load(R, f"{C}/sweeps_v2/sweep_imbalance_fixed_n3000.csv")
    INT, INTRA = load(R, f"{C}/sweeps_v2/sweep_inter_density_fixed.csv"), load(R, f"{C}/sweeps_v2/sweep_intra_density_fixed.csv")
    DPC = {k: load(R, f"{C}/sweeps_v2/sweep_{k}_fixed_DPC.csv") for k in ["overlap", "imbalance", "inter_density", "intra_density"]}
    DPCS = load(R, f"{C}/sweeps_v2/sweep_overlap_silhouette_DPC.csv")
    # older runs kept DPC in separate files; current sweeps include it. Append only when it is missing.
    with_dpc = lambda df, dp: pd.concat([df, dp]) if (df is not None and dp is not None and "DPC" not in set(df.algorithm)) else df
    E1F = load(R, f"{C}/exp1_difficulty_validation/exp1_results_fixed.csv")
    DEEP = load(R, "benchmarks/deep/results/deep_results.csv")
    REAL = load(R, "benchmarks/real-data/results/transfer_results.csv"); REALPA = load(R, "benchmarks/real-data/results/transfer_per_algo.csv")
    print("figures ->", out)

    # ------------------------------------------------------------------ factors_2d
    try:
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("mf", os.path.join(os.path.dirname(__file__), "make_factors_figure.py"))
        mf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mf); mf.OUT = out; mf.main()
    except Exception as e:
        print("  factors_2d skipped:", e)

    # ------------------------------------------------------------------ noise crossover (S and F), eta = 0.3
    if NOISE is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        d = NOISE[NOISE.value_eta == 0.3]
        for alg in ORDER: line(axes[0], d, "value_eta_int", alg)
        axes[0].set_title("Protocol S, $\\eta = 0.30$"); axes[0].set_xlabel("$\\eta_{int}$ (share of noise between clusters)"); axes[0].set_ylabel("inlier ARI")
        if EIF is not None:
            for alg in ORDER: line(axes[1], EIF, "value", alg)
            axes[1].set_title("Protocol F, $\\eta = 0.30$"); axes[1].set_xlabel("$\\eta_{int}$")
        p = d.groupby("value_eta_int")[["S_noise", "knn_noise_proximity", "D_dataset", "base_one_minus_silhouette", "S_separation"]].mean()
        for col, lab, c in [("S_noise", "$S_{noise}$", "#c0392b"), ("knn_noise_proximity", "noise proximity", "#e8743b"), ("D_dataset", "$D$", "k"),
                            ("base_one_minus_silhouette", "1 $-$ Silhouette (ref.)", "#6baed6"), ("S_separation", "$S_{sep}$", "#1f5fa8")]:
            axes[2].plot(p.index, p[col], marker="o", label=lab, color=c)
        axes[2].set_title("Profile along $\\eta_{int}$"); axes[2].set_xlabel("$\\eta_{int}$"); axes[2].legend(frameon=False)
        for ax in axes[:2]: ax.set_ylim(-0.05, 1.08)
        legend_families(axes[1] if EIF is not None else axes[0], ORDER, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5)
        save(fig, out, "noise_crossover_v2")

        # full grid, one panel per algorithm, lines = eta_int
        algs = [a for a in ORDER if a in NOISE.algorithm.unique()]
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharey=True)
        cmap = plt.cm.viridis(np.linspace(0, 1, 5))
        for ax, alg in zip(axes.ravel(), algs):
            dd = NOISE[NOISE.algorithm == alg]
            for ei, c in zip([0, .25, .5, .75, 1.0], cmap):
                g = dd[dd.value_eta_int == ei].groupby("value_eta").ARI.mean()
                ax.plot(g.index, g.values, marker="o", ms=4, color=c, label=f"$\\eta_{{int}}$ = {ei}")
            ax.set_title(alg); ax.set_xlabel("$\\eta$"); ax.set_ylim(-0.05, 1.08)
        axes[0, 0].set_ylabel("inlier ARI"); axes[1, 0].set_ylabel("inlier ARI")
        axes[0, 0].legend(frameon=False, fontsize=8)
        save(fig, out, "noise_grid_v2")

        # supplement: conventions
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
        for ax, (col, t) in zip(axes, [("ARI", "inlier ARI"), ("ARI_kept", "ARI on kept points"), ("ARI_full", "ARI with noise as a class")]):
            for alg in ORDER: line(ax, d, "value_eta_int", alg, y=col, band=False)
            ax.set_title(t); ax.set_xlabel("$\\eta_{int}$ ($\\eta = 0.30$, protocol S)")
        axes[0].set_ylabel("score"); legend_families(axes[1], ORDER, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5)
        save(fig, out, "noise_grid_conventions_S")

    # ------------------------------------------------------------------ overlap S / F + profile
    if OVF is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for alg in ORDER:
            if OVS is not None: line(axes[0], with_dpc(OVS, DPCS), "value", alg)
            line(axes[1], with_dpc(OVF, DPC["overlap"]), "value", alg)
        axes[0].set_title("Protocol S"); axes[1].set_title("Protocol F")
        for ax in axes[:2]: ax.set_xlabel("overlap $\\sigma$"); ax.set_ylabel("inlier ARI"); ax.set_ylim(-0.05, 1.08); ax.axhline(0.5, color="k", lw=.7, ls=":")
        p = OVF.groupby("value")[["S_separation", "S_boundary", "D_dataset", "knn_edge_impurity", "knn_fragmentation", "base_knn_cv_error"]].mean()
        for col, lab in [("S_separation", "$S_{sep}$"), ("S_boundary", "$S_{bnd}$"), ("D_dataset", "$D$"), ("knn_edge_impurity", "edge impurity"), ("knn_fragmentation", "fragmentation"), ("base_knn_cv_error", "$k$-NN CV error")]:
            axes[2].plot(p.index, p[col], marker="o", label=lab)
        axes[2].set_title("Profile along $\\sigma$"); axes[2].set_xlabel("overlap $\\sigma$"); axes[2].legend(frameon=False, fontsize=8)
        legend_families(axes[0], ORDER, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5)
        save(fig, out, "overlap_SF_v2")

    # ------------------------------------------------------------------ anisotropy S / F
    if E2 is not None and ANIF is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        aS = E2[E2.parameter == "anisotropy"]
        for alg in ORDER: line(axes[0], aS, "value", alg); line(axes[1], ANIF, "value", alg)
        axes[0].set_title("Protocol S"); axes[1].set_title("Protocol F")
        for ax in axes: ax.set_xlabel("anisotropy $\\alpha$"); ax.axhline(0.5, color="k", lw=.7, ls=":"); ax.set_ylim(-0.05, 1.08)
        axes[0].set_ylabel("inlier ARI"); legend_families(axes[0], ORDER, loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=5)
        save(fig, out, "anisotropy_SF_v2")

    # ------------------------------------------------------------------ dimension two regimes (+ supplement: Silhouette, K)
    if E2 is not None and DIMF is not None:
        dd = E2[E2.parameter == "dimensionality"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        for alg in ORDER: line(axes[0], dd, "value", alg); line(axes[1], DIMF, "value", alg)
        axes[0].set_title("$n = \\max(1000,\\, 50d)$"); axes[1].set_title("$n = 1000$ fixed")
        for ax in axes: ax.set_xlabel("dimension $d$"); ax.set_xscale("log"); ax.set_ylim(-0.05, 1.08)
        axes[0].set_ylabel("inlier ARI"); legend_families(axes[0], ORDER, loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=5)
        save(fig, out, "dimension_two_regimes_v2")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for alg in ORDER: line(axes[0], DIMF, "value", alg, y="Silhouette", band=False); line(axes[1], DIMF, "value", alg, y="n_clusters_found", band=False)
        axes[0].set_title("Silhouette of the output, $n$ fixed"); axes[1].set_title("$\\hat K$, $n$ fixed (true $K = 5$)"); axes[1].axhline(5, color="k", ls="--", lw=.8)
        for ax in axes: ax.set_xlabel("dimension $d$"); ax.set_xscale("log")
        legend_families(axes[0], ORDER, loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=5)
        save(fig, out, "dimension_silhouette_K_fixedn")

    # ------------------------------------------------------------------ density + imbalance (F), ARI and smallest-cluster F1
    if IMB is not None and INT is not None and INTRA is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
        panels = [("inter_density", INT, "$\\delta_{inter}$", "inter-cluster density heterogeneity"),
                  ("intra_density", INTRA, "$s_{intra}$ (core-dense)", "intra-cluster density profile"),
                  ("imbalance", IMB, "$\\rho_{imb}$", "size imbalance")]
        for j, (k, df, xl, t) in enumerate(panels):
            dfa = with_dpc(df, DPC[k])
            for alg in ORDER: line(axes[0, j], dfa, "value", alg); line(axes[1, j], dfa, "value", alg, y="min_cluster_f1")
            axes[0, j].set_title(t); axes[1, j].set_xlabel(xl)
            axes[0, j].set_ylim(-0.05, 1.08); axes[1, j].set_ylim(-0.05, 1.08)
        axes[0, 0].set_ylabel("inlier ARI"); axes[1, 0].set_ylabel("F1 of the smallest cluster")
        legend_families(axes[1, 1], ORDER, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5)
        save(fig, out, "density_imbalance_F_v2")
        if IMB3 is not None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
            for alg in ORDER: line(axes[0], IMB, "value", alg, y="min_cluster_f1"); line(axes[1], IMB3, "value", alg, y="min_cluster_f1")
            axes[0].set_title("$n = 1000$ (smallest cluster 180 $\\to$ 10)"); axes[1].set_title("$n = 3000$ (smallest cluster 540 $\\to$ 30)")
            for ax in axes: ax.set_xlabel("$\\rho_{imb}$")
            axes[0].set_ylabel("F1 of the smallest cluster"); legend_families(axes[0], ORDER, loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=5)
            save(fig, out, "imbalance_n3000")

    # ------------------------------------------------------------------ real data
    if REAL is not None and REALPA is not None:
        from scipy.stats import spearmanr
        df = REAL[REAL.relabel == False].copy() if "relabel" in REAL else REAL.copy()
        df["degen"] = df.dataset.map(REALPA.groupby("dataset").degenerate.mean())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        sc = axes[0].scatter(df.D_dataset, df.clean_median, s=50 + 700 * df.degen, c=df.knn_intrinsic_dim_twonn, cmap="viridis", edgecolor="k", lw=.6)
        texts = [axes[0].text(r.D_dataset, r.clean_median, r.dataset, fontsize=8) for _, r in df.iterrows()]
        try:   # repel overlapping labels when adjustText is installed (pip install adjustText); plain offsets otherwise
            from adjustText import adjust_text
            adjust_text(texts, ax=axes[0], arrowprops=dict(arrowstyle="-", color="grey", lw=.5), expand=(1.3, 1.6))
        except ImportError:
            for t in texts: t.set_position((t.get_position()[0] + 0.005, t.get_position()[1] + 0.003))
        rr = spearmanr(df.D_dataset, df.clean_median)
        axes[0].set_xlabel("$D$ (labels as given)"); axes[0].set_ylabel("median Silhouette of non-degenerate partitions")
        axes[0].set_title(f"$\\rho$ = {rr.statistic:.2f} (p = {rr.pvalue:.2f}); marker size = degenerate fraction", fontsize=10)
        plt.colorbar(sc, ax=axes[0], label="intrinsic dimension (TwoNN)")
        # right: datasets sorted by intrinsic dimension (lollipop), marker colour = degenerate fraction
        d2 = df.sort_values("knn_intrinsic_dim_twonn").reset_index(drop=True)
        cmap = plt.get_cmap("Reds"); norm = plt.Normalize(0, max(0.5, d2.degen.max()))
        axes[1].hlines(d2.index, 0, d2.knn_intrinsic_dim_twonn, color="#bbbbbb", lw=1.2)
        axes[1].scatter(d2.knn_intrinsic_dim_twonn, d2.index, s=110, c=cmap(norm(d2.degen)), edgecolor="k", lw=.6, zorder=3)
        for i, r in d2.iterrows():
            axes[1].text(r.knn_intrinsic_dim_twonn + 0.6, i, f"{r.degen:.2f}", va="center", fontsize=8)
        axes[1].set_yticks(d2.index); axes[1].set_yticklabels(d2.dataset, fontsize=8)
        axes[1].set_xlim(0, d2.knn_intrinsic_dim_twonn.max() * 1.15)
        r2 = spearmanr(df.knn_intrinsic_dim_twonn, df.degen)
        axes[1].set_xlabel("intrinsic dimension (TwoNN)")
        axes[1].set_title(f"degenerate fraction (label, colour) vs intrinsic dimension: $\\rho$ = {r2.statistic:.2f} (p = {r2.pvalue:.3f})", fontsize=10)
        axes[1].grid(axis="x", alpha=.3)
        save(fig, out, "transfer_real_v2")

    # ------------------------------------------------------------------ deep
    if DEEP is not None and E1F is not None:
        from scipy.stats import spearmanr
        seeds = [42, 123, 456, 789, 1011]
        E1F = E1F.copy(); E1F["idx"] = E1F.config_id * 5 + E1F.seed.map({s: i for i, s in enumerate(seeds)})
        prof = E1F.groupby("idx").D_dataset.first(); clas = E1F.groupby("idx").ARI.mean()
        D = DEEP.copy(); D["idx"] = D.dataset.str.extract(r"dataset_(\d+)_").astype(int)
        dm = D.groupby(["idx", "method"]).agg(ARI=("ARI", "mean"), K=("n_clusters_found", "mean")).reset_index()
        dm["D"] = dm.idx.map(prof)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for m in ["IDEC", "DCN", "DipDECK"]:
            g = dm[dm.method == m]; c, mk, _ = STYLE[m]
            axes[0].scatter(g.D, g.ARI, s=28, color=c, marker=mk, label=m, alpha=.85, edgecolor="k", lw=.3)
        axes[0].scatter(prof, clas, s=22, marker="x", color="k", label="classical pool, mean (F)", alpha=.6)
        axes[0].set_xlabel("$D$"); axes[0].set_ylabel("inlier ARI (mean of 5 runs)"); axes[0].legend(frameon=False)
        g = dm[dm.method == "DipDECK"]
        sc = axes[1].scatter(g.D, g.K, s=34, c=g.idx // 25, cmap="viridis", edgecolor="k", lw=.3)
        axes[1].axhline(5, ls="--", color="k", lw=.8); axes[1].set_xlabel("$D$"); axes[1].set_ylabel("DipDECK $\\hat K$ (mean of 5 runs)")
        axes[1].set_title(f"$\\rho(D, \\hat K)$ = {spearmanr(g.D, g.K).statistic:.2f}", fontsize=10)
        cb = plt.colorbar(sc, ax=axes[1], ticks=[0, 1, 2]); cb.ax.set_yticklabels(["easy", "medium", "hard"])
        save(fig, out, "deep_v2")


if __name__ == "__main__":
    main()
