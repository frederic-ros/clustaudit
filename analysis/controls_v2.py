#!/usr/bin/env python
"""
Controls for the fragile points of the paper, from result files.

    python analysis/controls_v2.py [--results-root .] [--out analysis/tables_paper]

1. Reference refinement (hard regime of the suite, protocol F): the pre-noise refinement run with
   k-NN (default), SVM, random forest, or not at all (raw generative labels). Per method: mean
   inlier ARI and the detectable / silent split under each reference, plus the fraction of labels
   each refinement changed.                                         -> tab_refinement_control.tex
2. Number of clusters (K = 3, 5, 8; protocol F, n = 1000, d = 20): the two emblematic findings
   (noise-placement crossover at eta = 0.3; imbalance: aggregate up, smallest cluster down).                                          -> tab_K_control.tex
"""
from __future__ import annotations
import argparse, glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ORDER = ["KMeans", "GMM", "Spectral", "Ward", "Average", "Complete", "DBSCAN", "HDBSCAN", "OPTICS", "DPC"]
NAMES = {"KMeans": "K-Means"}


def load(root, rel):
    f = sorted(glob.glob(os.path.join(root, rel))); return pd.read_csv(f[0], float_precision="round_trip") if f else None   # exact parsing: same doubles under every pandas version


def f2(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "--"
    t = f"{v:.{d}f}"
    return t[1:] if t.startswith("-") and float(t) == 0 else t   # no "-0.00"


def write(out, name, header, rows, colspec):
    txt = f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(out, name), "w").write(txt); print("  " + name)


def modes(d):
    deg = d.degenerate.astype(bool)
    return deg.mean(), ((~deg) & (d.ARI < 0.3)).mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-root", default="."); ap.add_argument("--out", default="analysis/tables_paper")
    a = ap.parse_args(); R, out = a.results_root, a.out; os.makedirs(out, exist_ok=True)
    E = "benchmarks/classical/results/exp1_difficulty_validation"; S = "benchmarks/classical/results/sweeps_v2"

    # ------------------------------------------------------------------ 1. refinement
    refs = {}
    raw = load(R, f"{E}/exp1_results_fixed_noprerefine.csv")
    if raw is not None:
        raw = raw[raw.difficulty == "hard"].copy(); raw["frac_relabelled"] = 0.0; refs["raw labels"] = raw
    knn = load(R, f"{E}/exp1_results_fixed.csv"); knn_dpc = load(R, f"{E}/exp1_results_fixed_DPC.csv")
    if knn is not None:   # the default reference: the suite's own result files
        if knn_dpc is not None and "DPC" not in knn.algorithm.unique(): knn = pd.concat([knn, knn_dpc], ignore_index=True)
        knn = knn[knn.difficulty == "hard"].copy()
        # exp1 records, in every row, the fraction of labels the pre-noise refinement changed
        refs["$k$-NN (default)"] = knn
    for clf, lab in (("svm", "SVM"), ("random_forest", "RF")):
        d = load(R, f"{E}/exp1_results_fixed_refine{clf}_hard.csv")
        if d is not None: refs[lab] = d
    if len(refs) >= 2:
        cols = list(refs)
        algs = [x for x in ORDER if all(x in refs[c].algorithm.unique() for c in cols)]
        rows = []
        for alg in algs:
            cells = [NAMES.get(alg, alg)]
            cells += [f2(refs[c][refs[c].algorithm == alg].ARI.mean()) for c in cols]
            cells += ["{:.2f}/{:.2f}".format(*modes(refs[c][refs[c].algorithm == alg])) for c in cols]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\midrule")
        fr = []
        for c in cols:
            d = refs[c]
            fr.append(f2(d.groupby(["config_id", "seed"]).frac_relabelled.first().mean()) if "frac_relabelled" in d else "--")
        rows.append("labels changed by the refinement & " + " & ".join(fr) + " & " * len(cols) + " \\\\")
        base = cols[1] if len(cols) > 1 else cols[0]
        rho = []
        for c in cols:
            x = [refs[base][refs[base].algorithm == g].ARI.mean() for g in algs]; y = [refs[c][refs[c].algorithm == g].ARI.mean() for g in algs]
            rho.append(f2(spearmanr(x, y).statistic))
        rows.append(f"rank agreement of the methods with {base} & " + " & ".join(rho) + " & " * len(cols) + " \\\\")
        head = (" & \\multicolumn{%d}{c}{mean inlier ARI, reference refined by} & \\multicolumn{%d}{c}{detectable / silent} \\\\\n"
                "\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}\nMethod & " % (len(cols), len(cols), len(cols) + 1, len(cols) + 2, 2 * len(cols) + 1)) + " & ".join(cols + cols)
        write(out, "tab_refinement_control.tex", head, rows, "@{}l" + "c" * (2 * len(cols)) + "@{}")

    # ------------------------------------------------------------------ 2. number of clusters
    Ks = [3, 5, 8]
    def sw(name, K):
        rel = f"{S}/{name}.csv" if K == 5 else f"{S}/{name}_K{K}.csv"
        d = load(R, rel)
        if K == 5 and d is not None:   # merge DPC run separately at K = 5
            dp = load(R, f"{S}/{name.replace('_fixed', '_fixed')}_DPC.csv")
            if dp is not None and "DPC" not in d.algorithm.unique(): d = pd.concat([d, dp])
        return d
    eta = {K: (load(R, f"{S}/sweep_eta_int_fixed_eta0.3.csv") if K == 5 else load(R, f"{S}/sweep_eta_int_fixed_eta0.3_K{K}.csv")) for K in Ks}
    imb = {K: sw("sweep_imbalance_fixed", K) for K in Ks}
    ov = {K: sw("sweep_overlap_fixed", K) for K in Ks}
    if all(eta[K] is not None for K in Ks):
        algs = [x for x in ORDER if all(x in eta[K].algorithm.unique() for K in Ks)]
        rows = []
        for alg in algs:
            cells = [NAMES.get(alg, alg)]
            for K in Ks:
                e = eta[K][eta[K].algorithm == alg]; cells.append(f"{f2(e[e.value == 0].ARI.mean())}/{f2(e[e.value == 1].ARI.mean())}")
            for K in Ks:
                d = imb[K]
                if d is None or alg not in d.algorithm.unique(): cells.append("--"); continue
                d = d[d.algorithm == alg]; lo, hi = d.value.min(), d.value.max()
                dA = d[d.value == hi].ARI.mean() - d[d.value == lo].ARI.mean(); dF = d[d.value == hi].min_cluster_f1.mean() - d[d.value == lo].min_cluster_f1.mean()
                cells.append(f"${dA:+.2f}$/${dF:+.2f}$")
            rows.append(" & ".join(cells) + " \\\\")
        kk = " & ".join(f"$K={K}$" for K in Ks)
        head = (" & \\multicolumn{3}{c}{noise placement, ARI distant/bridge} & \\multicolumn{3}{c}{imbalance $1 \\to 5$: $\\Delta$ARI/$\\Delta$F1$_{\\min}$} \\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nMethod & " + kk + " & " + kk)
        write(out, "tab_K_control.tex", head, rows, "@{}l" + "c" * 6 + "@{}")


def check_spectral_imbalance(R, seeds=(42, 123, 456, 789, 1011)):
    """Regenerate the imbalance sweep at rho = 5 and record what Spectral's best-matching cluster for the
    smallest reference cluster contains (inliers of that cluster, true-noise points, other inliers).
    Explains inlier ARI = 1.00 with smallest-cluster F1 = 0.50 (Table 8). Writes a CSV under R."""
    import sys; sys.path.insert(0, R)
    from clustaudit.audit.runner import _base_config, SWEEPS
    from clustaudit.core.pipeline import generate_dataset
    from clustaudit.audit.algorithms import run_pool
    from clustaudit.audit.evaluation import evaluate_partition
    rows = []
    for seed in seeds:
        cfg = _base_config(); SWEEPS["imbalance"](cfg, 5.0)
        st = generate_dataset(1000, 20, 5, structural=cfg["structural"], noise=cfg["noise"], ambiguity=cfg["ambiguity"], random_seed=seed)
        y = st.y; lab = np.asarray(run_pool(st.Z, seed, ["Spectral"], "fixed", n_clusters=5)["Spectral"])
        m = evaluate_partition(st.Z, y, lab)
        sizes = {k: int((y == k).sum()) for k in np.unique(y[y >= 0])}; small = min(sizes, key=sizes.get)
        best = np.bincount(lab[(y == small) & (lab >= 0)]).argmax(); mem = lab == best
        rows.append({"seed": seed, "inlier_ARI": m["ARI"], "small_size": sizes[small], "returned_size": int(mem.sum()),
                     "of_small": int((mem & (y == small)).sum()), "true_noise": int((mem & (y < 0)).sum()),
                     "other_inliers": int((mem & (y >= 0) & (y != small)).sum()),
                     "f1_all_points": m["min_cluster_f1"], "f1_inliers_only": m["min_cluster_f1_inlier"]})
    d = pd.DataFrame(rows); path = os.path.join(R, "benchmarks/classical/results/sweeps_v2/spectral_imbalance_check.csv")
    d.to_csv(path, index=False); print(d.round(3).to_string(index=False)); print("-> " + path)


if __name__ == "__main__":
    import sys
    if "--check-spectral" in sys.argv:
        r = sys.argv[sys.argv.index("--results-root") + 1] if "--results-root" in sys.argv else "."
        check_spectral_imbalance(r)
    else:
        main()
