#!/usr/bin/env python
"""
Three analyses that answer "what does the audit give that a benchmark does not", from existing result files.

    python analysis/audit_value_v2.py [--results-root .] [--out analysis/tables_paper]

1. Predictive value (out-of-generator).  For each method, its *operating envelope* is the set of
   (profile, ARI) pairs measured on our generator (suite + sweeps, protocol F).  We predict the
   method's ARI on datasets from MDCGen and HAWKS (never seen) by nearest neighbours in profile
   space, and compare the error with two benchmark-style predictors that use the same runs:
   the method's mean ARI on the suite (its "leaderboard score"), and a linear fit on the
   reference Silhouette.  Also: for every pair of methods, does the envelope predict which
   of the two is better on the external datasets?  -> tab_audit_predictive.tex
2. Paired interventions.  The sweeps use the same seeds at every level, so the generator's
   random draws are shared across levels (same centres, same base points).  The effect of
   moving a level is then a *paired* difference.  We compare the standard error of the paired
   effect with the one that independent generations would give, per method and sweep.
   -> tab_paired_effects.tex
3. Threshold conventions.  Critical levels at ARI cut-offs 0.4 / 0.5 / 0.6 and failure modes
   at silent thresholds 0.2 / 0.3 / 0.4: does the ordering of the methods change?
   -> tab_threshold_conventions.tex
"""
from __future__ import annotations
import argparse, glob, os, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

ORDER = ["KMeans", "GMM", "Spectral", "Ward", "Average", "DBSCAN", "HDBSCAN", "OPTICS", "DPC"]
NAMES = {"KMeans": "K-Means"}
PROF = ["S_separation", "S_boundary", "S_shape", "S_noise", "knn_edge_impurity", "knn_ambiguous_fraction", "knn_mean_impurity",
        "knn_fragmentation", "knn_density_heterogeneity", "knn_intra_density_spread", "knn_noise_fraction", "knn_noise_proximity"]


def load(root, rel):
    f = sorted(glob.glob(os.path.join(root, rel))); return pd.read_csv(f[0], float_precision="round_trip") if f else None   # exact parsing: same doubles under every pandas version


def with_dpc(df, dp):
    """Older runs kept DPC in a separate file; current sweeps include it. Append the DPC file only when DPC is missing."""
    if df is None: return dp
    return pd.concat([df, dp], ignore_index=True) if (dp is not None and "DPC" not in set(df.algorithm)) else df


def cat(*dfs):
    return pd.concat([d for d in dfs if d is not None], ignore_index=True)


def write(out, name, header, rows, colspec):
    txt = f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(out, name), "w").write(txt); print("  " + name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-root", default="."); ap.add_argument("--out", default="analysis/tables_paper")
    a = ap.parse_args(); R, out = a.results_root, a.out; os.makedirs(out, exist_ok=True)
    C = "benchmarks/classical/results"
    suite = with_dpc(load(R, f"{C}/exp1_difficulty_validation/exp1_results_fixed.csv"), load(R, f"{C}/exp1_difficulty_validation/exp1_results_fixed_DPC.csv"))
    sweeps = cat(*[with_dpc(load(R, f"{C}/sweeps_v2/sweep_{k}_fixed.csv"), load(R, f"{C}/sweeps_v2/sweep_{k}_fixed_DPC.csv")) for k in ["overlap", "imbalance", "inter_density", "intra_density", "connectivity"]],
                 load(R, f"{C}/sweeps_v2/sweep_eta_int_fixed_eta0.3.csv"), load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_fixed_anisotropy.csv"))
    XG = load(R, "benchmarks/cross-generator/results/crossgen_results.csv")
    for d in (suite, sweeps, XG):
        d["knn_noise_proximity"] = d["knn_noise_proximity"].fillna(0.0)

    # ------------------------------------------------------------- 1. predictive value
    if XG is not None:
        xg = XG[XG.protocol == "fixed"]; rows = []; summary = {}
        per_method = {}
        for alg in ORDER:
            tr = cat(suite[suite.algorithm == alg], sweeps[sweeps.algorithm == alg]).dropna(subset=["ARI"])
            te = xg[xg.algorithm == alg].dropna(subset=["ARI"])
            if len(tr) < 20 or len(te) < 5: continue
            sc = StandardScaler().fit(tr[PROF]); knn = KNeighborsRegressor(n_neighbors=7, weights="distance").fit(sc.transform(tr[PROF]), tr.ARI)
            p_env = knn.predict(sc.transform(te[PROF]))
            p_bench = np.full(len(te), suite[suite.algorithm == alg].ARI.mean())
            lin = LinearRegression().fit(tr[["base_one_minus_silhouette"]], tr.ARI); p_sil = np.clip(lin.predict(te[["base_one_minus_silhouette"]]), 0, 1)
            mae = lambda p: np.mean(np.abs(p - te.ARI.values))
            per_method[alg] = dict(env=p_env, bench=p_bench, sil=p_sil, true=te.ARI.values, keys=list(zip(te.generator, te.native_param, te.seed)))
            rows.append(f"{NAMES.get(alg, alg)} & {len(te)} & {mae(p_bench):.3f} & {mae(p_sil):.3f} & {mae(p_env):.3f} \\\\")
        # pairwise "which method is better on this external dataset" from each predictor
        # uncertainty of the gains: bootstrap over external datasets (resampling dataset keys, so that the
        # nine methods' errors on one dataset move together), 2000 draws
        keys = sorted(set(k for v in per_method.values() for k in v["keys"])); rng = np.random.RandomState(0)
        err = {name: {} for name in ("bench", "sil", "env")}
        for alg, v in per_method.items():
            for name in err:
                for k, e in zip(v["keys"], np.abs(v[name] - v["true"])): err[name].setdefault(k, []).append(e)
        per_key = {name: np.array([np.mean(err[name][k]) for k in keys]) for name in err}
        boots = {name: [] for name in ("bench", "sil", "env")}; gains = {"env-bench": [], "env-sil": []}
        for _ in range(2000):
            idx = rng.randint(0, len(keys), len(keys))
            m = {name: per_key[name][idx].mean() for name in per_key}
            for name in boots: boots[name].append(m[name])
            gains["env-bench"].append(m["bench"] - m["env"]); gains["env-sil"].append(m["sil"] - m["env"])
        ci = lambda x: f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]"
        rows.append("\\midrule")
        rows.append(f"mean over the {len(keys)} external datasets & {len(keys)} & {per_key['bench'].mean():.3f} {ci(boots['bench'])} & {per_key['sil'].mean():.3f} {ci(boots['sil'])} & {per_key['env'].mean():.3f} {ci(boots['env'])} \\\\")
        rows.append(f"gain of the envelope [95\\% CI] & & {np.mean(gains['env-bench']):.3f} {ci(gains['env-bench'])} & {np.mean(gains['env-sil']):.3f} {ci(gains['env-sil'])} & \\\\")
        summary = {"mae_bench": float(per_key["bench"].mean()), "mae_sil": float(per_key["sil"].mean()), "mae_env": float(per_key["env"].mean()),
                   "gain_vs_bench": float(np.mean(gains["env-bench"])), "gain_vs_sil": float(np.mean(gains["env-sil"])),
                   "gain_vs_bench_lo": float(np.percentile(gains["env-bench"], 2.5)), "gain_vs_sil_lo": float(np.percentile(gains["env-sil"], 2.5)),
                   "gain_vs_bench_hi": float(np.percentile(gains["env-bench"], 97.5)), "gain_vs_sil_hi": float(np.percentile(gains["env-sil"], 97.5))}
        # the same, per generator: the bootstrap above is conditional on this collection of two generators
        for gen in ("mdcgen", "hawks"):
            kk = [i for i, k in enumerate(keys) if k[0] == gen]
            pk = {name: per_key[name][kk] for name in per_key}; gb, gs = [], []
            for _ in range(2000):
                idx = rng.randint(0, len(kk), len(kk)); m = {name: pk[name][idx].mean() for name in pk}
                gb.append(m["bench"] - m["env"]); gs.append(m["sil"] - m["env"])
            summary[f"{gen}_gain_vs_bench"] = float(np.mean(gb)); summary[f"{gen}_gain_vs_sil"] = float(np.mean(gs))
            summary[f"{gen}_gain_vs_sil_lo"] = float(np.percentile(gs, 2.5)); summary[f"{gen}_gain_vs_sil_hi"] = float(np.percentile(gs, 97.5))
            summary[f"{gen}_gain_vs_bench_lo"] = float(np.percentile(gb, 2.5)); summary[f"{gen}_gain_vs_bench_hi"] = float(np.percentile(gb, 97.5))
            label = {"mdcgen": "MDCGen", "hawks": "HAWKS"}[gen]
            rows.append(f"\\quad {label} ({len(kk)}): MAE / gain & & {pk['bench'].mean():.3f} / {np.mean(gb):.3f} {ci(gb)} & {pk['sil'].mean():.3f} / {np.mean(gs):.3f} {ci(gs)} & {pk['env'].mean():.3f} \\\\")
        algs = list(per_method)
        for label, close in (("all pairs", False), ("close pairs", True)):
            wins = {"bench": [0, 0], "sil": [0, 0], "env": [0, 0]}
            for a1, a2 in itertools.combinations(algs, 2):
                if close and abs(per_method[a1]["bench"][0] - per_method[a2]["bench"][0]) >= 0.1: continue
                k1 = {k: i for i, k in enumerate(per_method[a1]["keys"])}
                for j, k in enumerate(per_method[a2]["keys"]):
                    if k not in k1: continue
                    i = k1[k]; t = per_method[a1]["true"][i] - per_method[a2]["true"][j]
                    if abs(t) < 0.05: continue
                    for name in wins:
                        pred = per_method[a1][name][i] - per_method[a2][name][j]
                        wins[name][0] += int(np.sign(pred) == np.sign(t)); wins[name][1] += 1
            acc = {k: v[0] / max(v[1], 1) for k, v in wins.items()}
            if close: summary.update({"close_pairs_env": acc["env"], "close_pairs_sil": acc["sil"], "close_pairs_bench": acc["bench"]})
            rows.append(f"ordering accuracy, {label} ({wins['env'][1]}) & & {acc['bench']:.2f} & {acc['sil']:.2f} & {acc['env']:.2f} \\\\")
        sdir = os.path.join(R, "benchmarks/cross-generator/results"); os.makedirs(sdir, exist_ok=True)
        pd.Series(summary).to_csv(os.path.join(sdir, "predictive_summary.csv"), header=["value"],   # read by audit_paper_claims_v2.py
                                  float_format="%.10g")   # ten digits: identical files across library versions
        write(out, "tab_audit_predictive.tex", "Method & $N_{\\text{ext}}$ & \\shortstack{suite mean\\\\(leaderboard)} & \\shortstack{Silhouette\\\\fit} & \\shortstack{operating\\\\envelope}", rows, "@{}>{\\raggedright\\arraybackslash}p{4.2cm}cccc@{}")

    # ------------------------------------------------------------- 2. paired vs independent effects
    rows, prat = [], []
    for key, df in [("overlap $\\sigma$", load(R, f"{C}/sweeps_v2/sweep_overlap_fixed.csv")), ("anisotropy $\\alpha$", load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_fixed_anisotropy.csv")),
                    ("inter-density $\\delta$", load(R, f"{C}/sweeps_v2/sweep_inter_density_fixed.csv")), ("imbalance $\\rho$", load(R, f"{C}/sweeps_v2/sweep_imbalance_fixed.csv")),
                    ("noise placement $\\eta_{\\text{int}}$", load(R, f"{C}/sweeps_v2/sweep_eta_int_fixed_eta0.3.csv"))]:
        if df is None: continue
        ratios = {}
        for alg, g in df.groupby("algorithm"):
            piv = g.pivot_table(index="seed", columns="value", values="ARI"); vals = sorted(piv.columns); rs = []
            for v0, v1 in zip(vals[:-1], vals[1:]):
                d = (piv[v1] - piv[v0]).dropna()
                if len(d) < 3 or d.std() == 0 and piv[v0].std() == 0: continue
                se_pair = d.std(ddof=1) / np.sqrt(len(d)); se_ind = np.sqrt(piv[v0].var(ddof=1) / len(d) + piv[v1].var(ddof=1) / len(d))
                if se_ind > 0: rs.append(se_pair / se_ind)
            if rs: ratios[alg] = np.median(rs); prat.append({"sweep": key, "algorithm": alg, "ratio": ratios[alg]})
        if ratios:
            rows.append(key + " & " + " & ".join(f"{ratios[a]:.2f}" if a in ratios else "--" for a in ORDER) + " \\\\")
    if prat:   # read by audit_paper_claims_v2.py
        pd.DataFrame(prat).to_csv(os.path.join(R, f"{C}/sweeps_v2/paired_ratios.csv"), index=False, float_format="%.10g")
    if rows:
        write(out, "tab_paired_effects.tex", "Sweep & " + " & ".join(NAMES.get(a, a) for a in ORDER), rows, "@{}l" + "c" * len(ORDER) + "@{}")

    # ------------------------------------------------------------- 3. threshold conventions
    def crit(df, thr):
        res = {}
        for alg, g in df.groupby("algorithm"):
            m = g.groupby("value").ARI.mean(); b = m[m < thr]; res[alg] = b.index.min() if len(b) else np.inf
        return res
    rows = []
    for key, df in [("overlap $\\sigma_{\\text{crit}}$", load(R, f"{C}/sweeps_v2/sweep_overlap_fixed.csv")), ("anisotropy $\\alpha_{\\text{crit}}$", load(R, f"{C}/exp2_sensitivity_analysis/exp2_results_fixed_anisotropy.csv"))]:
        if df is None: continue
        c = {t: crit(df, t) for t in (0.4, 0.5, 0.6)}; algs = [a for a in ORDER if a in c[0.5]]
        fmt = lambda v: "$>$max" if np.isinf(v) else f"{v:g}"
        for t in (0.4, 0.5, 0.6):
            rows.append(f"{key}, cut-off {t} & " + " & ".join(fmt(c[t][a]) for a in algs) + " \\\\")
        r45 = spearmanr([c[0.4][a] if np.isfinite(c[0.4][a]) else 99 for a in algs], [c[0.5][a] if np.isfinite(c[0.5][a]) else 99 for a in algs]).statistic
        r56 = spearmanr([c[0.5][a] if np.isfinite(c[0.5][a]) else 99 for a in algs], [c[0.6][a] if np.isfinite(c[0.6][a]) else 99 for a in algs]).statistic
        rows.append(f"\\multicolumn{{{len(algs)+1}}}{{l}}{{\\quad ordering of the methods: $\\rho(0.4, 0.5) = {r45:.2f}$, $\\rho(0.5, 0.6) = {r56:.2f}$}} \\\\")
    if suite is not None and "degenerate" in suite:
        hard = suite[suite.difficulty == "hard"]; algs = [a for a in ORDER if a in hard.algorithm.unique()]
        for t in (0.2, 0.3, 0.4):
            cells = []
            for a_ in algs:
                g = hard[hard.algorithm == a_]; det = g.degenerate.astype(bool).mean(); sil = ((~g.degenerate.astype(bool)) & (g.ARI < t)).mean()
                cells.append(f"{det:.2f}/{sil:.2f}")
            rows.append(f"hard regime, silent threshold {t}: detectable/silent & " + " & ".join(cells) + " \\\\")
    if rows:
        write(out, "tab_threshold_conventions.tex", "Convention & " + " & ".join(NAMES.get(a, a) for a in ORDER), rows, "@{}l" + "c" * len(ORDER) + "@{}")


if __name__ == "__main__":
    main()
