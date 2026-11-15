#!/usr/bin/env python
"""
Re-derive every number quoted in the paper from the result CSVs.

    python analysis/audit_paper_claims_v2.py [--results-root .]

Each claim is (label, value stated in the paper, function computing it from the
CSVs, tolerance). The script prints a table and exits 1 if any claim fails, so
it can run as a test before submission. Add a claim whenever a number enters
the text.
"""
from __future__ import annotations
import argparse, glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

CLAIMS = []


def claim(label, stated, tol=0.015):
    def deco(fn):
        CLAIMS.append((label, stated, fn, tol)); return fn
    return deco


def ds_level(df, cols=("ARI",)):
    """Dataset-level means over the paper's suite pool (DPC is reported separately)."""
    df = df[df.algorithm != "DPC"] if "algorithm" in df else df
    key = ["config_id", "seed"] if "config_id" in df else ["value", "seed"]
    return df.groupby(key).agg({**{c: "mean" for c in cols}, "D_dataset": "first"})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-root", default=".")
    a = ap.parse_args(); R = a.results_root
    C = os.path.join(R, "benchmarks/classical/results")
    def load(p):
        f = glob.glob(os.path.join(R, p)); return pd.read_csv(f[0], float_precision="round_trip") if f else None   # exact parsing: same doubles under every pandas version

    E1F = load(f"{C}/exp1_difficulty_validation/exp1_results_fixed.csv")
    E1S = load(f"{C}/exp1_difficulty_validation/exp1_results.csv")
    NOISE = load(f"{C}/exp2_sensitivity_analysis/exp2_results_silhouette_noise.csv")
    DIMF = load(f"{C}/exp2_sensitivity_analysis/exp2_results_silhouette_fixedn.csv")
    DIMP = load(f"{C}/exp2_sensitivity_analysis/exp2_results.csv")
    ANIF = load(f"{C}/exp2_sensitivity_analysis/exp2_results_fixed_anisotropy.csv")
    OVF = load(f"{C}/sweeps_v2/sweep_overlap_fixed.csv")
    IMBF = load(f"{C}/sweeps_v2/sweep_imbalance_fixed.csv")
    INTF = load(f"{C}/sweeps_v2/sweep_inter_density_fixed.csv")
    DEEP = load("benchmarks/deep/results/deep_results.csv")
    REAL = load("benchmarks/real-data/results/transfer_results.csv")   # labels as given (not the _relabel file)
    XG = load("benchmarks/cross-generator/results/crossgen_results.csv")

    # ---------------- Q1 / validation ----------------
    @claim("r(D, mean ARI) suite, protocol F", -0.92)
    def _(): m = ds_level(E1F); return pearsonr(m.D_dataset, m.ARI)[0]
    @claim("r(D, mean ARI) suite, protocol S", -0.92)
    def _(): m = ds_level(E1S); return pearsonr(m.D_dataset, m.ARI)[0]
    @claim("rho(D, mean ARI) suite, F", -0.97)
    def _(): m = ds_level(E1F); return spearmanr(m.D_dataset, m.ARI).statistic

    # ---------------- Q3 noise crossover (S, eta=0.3) ----------------
    def noise_at(alg, ei, eta=0.3):
        d = NOISE[(NOISE.value_eta == eta) & (NOISE.value_eta_int == ei) & (NOISE.algorithm == alg)]; return d.ARI.mean()
    for alg, lo, hi in [("KMeans", 0.00, 0.77), ("Ward", 0.00, 0.75), ("GMM", 0.00, 0.77), ("Average", 0.00, 0.70),
                        ("HDBSCAN", 1.00, 0.47), ("OPTICS", 0.94, 0.00)]:
        claim(f"noise eta=0.3 {alg} ARI at eta_int=0", lo)(lambda alg=alg: noise_at(alg, 0.0))
        claim(f"noise eta=0.3 {alg} ARI at eta_int=1", hi)(lambda alg=alg: noise_at(alg, 1.0))
    claim("noise eta=0.3: DBSCAN degenerate in every run for eta_int <= 0.5, not at 0.75", 1.0, 0.0)(lambda: float(
        NOISE[(NOISE.algorithm == "DBSCAN") & (NOISE.value_eta == 0.3) & (NOISE.value_eta_int <= 0.5)].degenerate.astype(bool).all()
        and not NOISE[(NOISE.algorithm == "DBSCAN") & (NOISE.value_eta == 0.3) & (NOISE.value_eta_int == 0.75)].degenerate.astype(bool).all()))
    def _spreads():
        m = NOISE.groupby(["algorithm", "value_eta", "value_eta_int"]).ARI.mean()
        return m.groupby(["algorithm", "value_eta"]).agg(lambda x: x.max() - x.min()).groupby("algorithm").max()
    claim("noise grid: methods whose mean ARI moves by > 0.8 across eta_int at some eta", 5, 0)(lambda: float((_spreads() > 0.8).sum()))
    claim("noise grid: methods moving by > 0.5 (all but Spectral)", 7, 0)(lambda: float((_spreads() > 0.5).sum()))
    claim("noise grid: Spectral largest move", 0.37, 0.005)(lambda: _spreads()["Spectral"])
    @claim("noise eta=0.3 Spectral at eta_int=0", 0.84)
    def _(): return noise_at("Spectral", 0.0)
    @claim("noise eta=0.3: S_noise at eta_int=0", 0.45)
    def _(): return NOISE[(NOISE.value_eta == 0.3) & (NOISE.value_eta_int == 0)].S_noise.mean()
    @claim("noise eta=0.3: S_noise at eta_int=1", 0.85)
    def _(): return NOISE[(NOISE.value_eta == 0.3) & (NOISE.value_eta_int == 1)].S_noise.mean()
    @claim("noise eta=0.3: noise_proximity at eta_int=1", 0.44)
    def _(): return NOISE[(NOISE.value_eta == 0.3) & (NOISE.value_eta_int == 1)].knn_noise_proximity.mean()
    @claim("noise eta=0.3: HDBSCAN ARI_kept at eta_int=0", 1.00)
    def _(): return NOISE[(NOISE.value_eta == 0.3) & (NOISE.value_eta_int == 0) & (NOISE.algorithm == "HDBSCAN")].ARI_kept.mean()

    # ---------------- Q3 noise crossover control (F, eta=0.3) and imbalance at n=3000 ----------------
    EIF = load(f"{C}/sweeps_v2/sweep_eta_int_fixed_eta0.3.csv")
    if EIF is not None:
        def ei(alg, v): return EIF[(EIF.algorithm == alg) & (EIF.value == v)].ARI.mean()
        for alg, lo, hi in [("KMeans", 0.04, 1.00), ("Ward", 0.00, 0.97), ("GMM", 0.00, 0.92), ("HDBSCAN", 1.00, 0.34),
                            ("OPTICS", 0.91, 0.00), ("Spectral", 1.00, 0.99), ("DPC", 1.00, 0.81)]:
            claim(f"crossover F {alg} eta_int=0", lo)(lambda alg=alg: ei(alg, 0.0))
            claim(f"crossover F {alg} eta_int=1", hi)(lambda alg=alg: ei(alg, 1.0))
    IMB3 = load(f"{C}/sweeps_v2/sweep_imbalance_fixed_n3000.csv")
    if IMB3 is not None:
        for alg, lo, hi in [("KMeans", 0.95, 0.02), ("Ward", 0.82, 0.02), ("Spectral", 0.96, 0.19)]:
            claim(f"imbalance n=3000 {alg} minF1 rho=1", lo)(lambda alg=alg: IMB3[(IMB3.algorithm == alg) & (IMB3.value == 1.0)].min_cluster_f1.mean())
            claim(f"imbalance n=3000 {alg} minF1 rho=5", hi)(lambda alg=alg: IMB3[(IMB3.algorithm == alg) & (IMB3.value == 5.0)].min_cluster_f1.mean())

    # ---------------- label smoothing control ----------------
    NOPRE = load(f"{C}/exp1_difficulty_validation/exp1_results_fixed_noprerefine.csv")
    if NOPRE is not None:
        claim("no-prerefine: r(D, mean ARI)", -0.93)(lambda: pearsonr(*ds_level(NOPRE)[["D_dataset", "ARI"]].values.T)[0])
        claim("no-prerefine: Ward hard ARI", 0.12)(lambda: NOPRE[(NOPRE.algorithm == "Ward") & (NOPRE.difficulty == "hard")].ARI.mean())

    # ---------------- Q3 overlap (F) ----------------
    def ov(alg, v): return OVF[(OVF.algorithm == alg) & (OVF.value == v)].ARI.mean()
    claim("overlap F KMeans sigma=0.4", 0.78)(lambda: ov("KMeans", 0.4))
    claim("overlap F KMeans sigma=1.0", 0.38)(lambda: ov("KMeans", 1.0))
    claim("overlap F Spectral sigma=0.6", 0.94)(lambda: ov("Spectral", 0.6))
    claim("overlap F Spectral sigma=1.0", 0.52)(lambda: ov("Spectral", 1.0))
    claim("overlap F DBSCAN degenerate at sigma>=0.4", 1.0)(lambda: OVF[(OVF.algorithm == "DBSCAN") & (OVF.value >= 0.4)].degenerate.mean())
    claim("overlap F: five seeds per cell", 5, 0)(lambda: OVF.groupby(["algorithm", "value"]).seed.nunique().min())
    claim("overlap F HDBSCAN degenerate at sigma=0.4", 0.8, 0.01)(lambda: OVF[(OVF.algorithm == "HDBSCAN") & (OVF.value == 0.4)].degenerate.mean())
    claim("overlap F OPTICS degenerate at sigma>=0.4", 1.0, 0.0)(lambda: OVF[(OVF.algorithm == "OPTICS") & (OVF.value >= 0.4)].degenerate.mean())
    _ovp = lambda c, v: OVF.groupby(["value", "seed"]).first().groupby("value")[c].mean()[v]
    claim("overlap: kNN CV error at sigma=0.4", 0.001, 0.001)(lambda: _ovp("base_knn_cv_error", 0.4))
    claim("overlap: edge impurity at sigma=0.4", 0.009, 0.002)(lambda: _ovp("knn_edge_impurity", 0.4))
    claim("overlap: edge impurity at sigma=1.4", 0.36, 0.01)(lambda: _ovp("knn_edge_impurity", 1.4))
    claim("overlap: fragmentation at sigma=0.2", 0.08, 0.01)(lambda: _ovp("knn_fragmentation", 0.2))
    claim("overlap: fragmentation at sigma=1.4", 0.27, 0.01)(lambda: _ovp("knn_fragmentation", 1.4))
    claim("overlap: S_sep at sigma=0.6..1.0", 0.62, 0.01)(lambda: _ovp("S_separation", 0.8))
    claim("overlap: kNN cv error at sigma=0.4", 0.002, 0.005)(lambda: OVF[OVF.value == 0.4].base_knn_cv_error.mean())
    claim("overlap: edge impurity at sigma=0.4", 0.011, 0.005)(lambda: OVF[OVF.value == 0.4].knn_edge_impurity.mean())

    # ---------------- Q3 anisotropy (F) ----------------
    def acrit(alg):
        m = ANIF[ANIF.algorithm == alg].groupby("value").ARI.mean(); b = m[m < 0.5]; return b.index.min() if len(b) else np.inf
    for alg, v in [("Spectral", 10), ("KMeans", 8), ("Ward", 6), ("GMM", 4), ("HDBSCAN", 3), ("DBSCAN", 2), ("OPTICS", 1.5)]:
        claim(f"alpha_crit F {alg}", v, 0.01)(lambda alg=alg: acrit(alg))

    # ---------------- Q3 dimension ----------------
    def dim(df, alg, d): return df[(df.algorithm == alg) & (df.value == d)].ARI.mean()
    DIMN = DIMP[DIMP.parameter == "dimensionality"] if DIMP is not None else None
    claim("dimension fixed-n KMeans d=30", 0.95, 0.005)(lambda: dim(DIMF, "KMeans", 30))
    claim("dimension fixed-n KMeans d=100", 0.16, 0.005)(lambda: dim(DIMF, "KMeans", 100))
    claim("dimension fixed-n Ward d=30", 0.73, 0.005)(lambda: dim(DIMF, "Ward", 30))
    claim("dimension fixed-n Ward d=70", 0.00, 0.005)(lambda: dim(DIMF, "Ward", 70))
    claim("dimension fixed-n Ward d=100", 0.00, 0.005)(lambda: dim(DIMF, "Ward", 100))
    claim("dimension fixed-n Spectral min over d>=30", 0.93, 0.005)(lambda: min(dim(DIMF, "Spectral", d) for d in (30, 50, 70, 100)))
    claim("dimension n∝d KMeans min over d>=10", 0.97, 0.005)(lambda: min(dim(DIMN, "KMeans", d) for d in (10, 15, 20, 30, 50, 70, 100)))
    claim("dimension n∝d Spectral min over d>=10", 0.80, 0.005)(lambda: min(dim(DIMN, "Spectral", d) for d in (10, 15, 20, 30, 50, 70, 100)))
    claim("dimension n∝d Ward min over d>=10", 0.78, 0.005)(lambda: min(dim(DIMN, "Ward", d) for d in (10, 15, 20, 30, 50, 70, 100)))
    claim("dimension GMM max over d>=70, both regimes", 0.13, 0.005)(lambda: max(dim(D_, "GMM", d) for D_ in (DIMF, DIMN) for d in (70, 100)))
    claim("dimension TwoNN at d=100, n fixed", 54, 0.5)(lambda: DIMF[DIMF.value == 100].knn_intrinsic_dim_twonn.mean())

    # ---------------- Q3 density / imbalance (F) ----------------
    def at(df, alg, v, col="ARI"): return df[(df.algorithm == alg) & (df.value == v)][col].mean()
    claim("inter-density F DBSCAN delta=0.75", 0.00)(lambda: at(INTF, "DBSCAN", 0.75))
    claim("inter-density F HDBSCAN delta=1", 0.66)(lambda: at(INTF, "HDBSCAN", 1.0))
    claim("imbalance F DBSCAN ARI rho=5", 0.98)(lambda: at(IMBF, "DBSCAN", 5.0))
    claim("imbalance F DBSCAN minF1 rho=5", 0.00)(lambda: at(IMBF, "DBSCAN", 5.0, "min_cluster_f1"))
    claim("imbalance F KMeans minF1 rho=1", 0.69)(lambda: at(IMBF, "KMeans", 1.0, "min_cluster_f1"))
    claim("imbalance F KMeans minF1 rho=5", 0.06)(lambda: at(IMBF, "KMeans", 5.0, "min_cluster_f1"))

    # ---------------- Q3 failure modes (suite, F) ----------------
    def fm(alg, reg, mode): d = E1F[(E1F.algorithm == alg) & (E1F.difficulty == reg)]; return (d.failure_mode == mode).mean()
    claim("failure hard DBSCAN declared", 1.00)(lambda: fm("DBSCAN", "hard", "declared"))
    claim("failure hard Ward silent", 0.92)(lambda: fm("Ward", "hard", "silent"))
    claim("failure hard KMeans ok", 0.76)(lambda: fm("KMeans", "hard", "ok"))
    claim("failure hard Spectral ok", 0.88)(lambda: fm("Spectral", "hard", "ok"))
    claim("failure easy Average silent", 0.80)(lambda: fm("Average", "easy", "silent"))

    # ---------------- Q3 divergence table (F) ----------------
    def rr(alg, col):
        d = E1F[E1F.algorithm == alg].dropna(subset=[col]); return pearsonr(d.D_dataset, d[col])[0]
    for alg, v in [("Average", -0.33), ("Ward", -0.89), ("DBSCAN", -0.88), ("HDBSCAN", -0.92), ("KMeans", -0.73)]:
        claim(f"r(D, ARI) {alg} F", v)(lambda alg=alg: rr(alg, "ARI"))
    for alg, v in [("Average", -0.92), ("Ward", 0.17), ("GMM", -0.09), ("KMeans", -0.65)]:
        claim(f"r(D, Sil) {alg} F", v)(lambda alg=alg: rr(alg, "Silhouette"))
    # tuning protocol (Section 6.3): runs with no admissible configuration are degenerate rows
    E1D = load(f"{C}/exp1_difficulty_validation/exp1_results_dbcv.csv")
    def _deg(d, alg): return d[d.algorithm == alg].degenerate.astype(bool).mean()
    def _r(d, alg): g = d[d.algorithm == alg].dropna(subset=["ARI"]); return pearsonr(g.D_dataset, g.ARI)[0]
    def _noadm(d, alg): return float((d[d.algorithm == alg].get("run_status", pd.Series(dtype=str)) == "no_admissible_configuration").sum())
    claim("DBSCAN degenerate fraction under S", 0.84, 0.005)(lambda: _deg(E1S, "DBSCAN"))
    claim("DBSCAN r(D,ARI) under S", -0.38, 0.005)(lambda: _r(E1S, "DBSCAN"))
    if E1D is not None:
        claim("DBSCAN degenerate fraction under D", 0.48, 0.005)(lambda: _deg(E1D, "DBSCAN"))
        claim("DBSCAN r(D,ARI) under D", -0.91, 0.005)(lambda: _r(E1D, "DBSCAN"))
        claim("DBSCAN runs without admissible configuration, S and D (all hard)", 21, 0)(
            lambda: _noadm(E1S, "DBSCAN") if _noadm(E1S, "DBSCAN") == _noadm(E1D, "DBSCAN") ==
            float(((E1S.algorithm == "DBSCAN") & (E1S.get("run_status") == "no_admissible_configuration") & (E1S.difficulty == "hard")).sum()) else -1)
        claim("r(D, mean ARI) suite, protocol D", -0.92, 0.005)(lambda: pearsonr(*ds_level(E1D)[["D_dataset", "ARI"]].values.T)[0])
    claim("DBSCAN r(D,ARI) under F", -0.88, 0.005)(lambda: _r(E1F, "DBSCAN"))
    for alg, lo, hi in [("HDBSCAN", 0.43, 0.47), ("OPTICS", 0.67, 0.68)]:   # "about half", "two thirds", under all three
        claim(f"{alg} degenerate fraction, min over S/D/F", lo, 0.005)(lambda alg=alg: min(_deg(d, alg) for d in (E1S, E1D, E1F) if d is not None))
        claim(f"{alg} degenerate fraction, max over S/D/F", hi, 0.005)(lambda alg=alg: max(_deg(d, alg) for d in (E1S, E1D, E1F) if d is not None))

    # ---------------- Q4 real data ----------------
    if REAL is not None:
        raw = REAL[REAL.relabel == False] if "relabel" in REAL else REAL
        claim("real: rho(D, clean_median) raw labels", -0.41, 0.02)(lambda raw=raw: spearmanr(raw.D_dataset, raw.clean_median).statistic)

    # ---------------- Q5 deep ----------------
    if DEEP is not None and E1F is not None:
        seeds = [42, 123, 456, 789, 1011]
        E1F["idx"] = E1F.config_id * 5 + E1F.seed.map({s: i for i, s in enumerate(seeds)})
        prof = E1F.groupby("idx").D_dataset.first()
        DEEP["idx"] = DEEP.dataset.str.extract(r"dataset_(\d+)_").astype(int)
        dm = DEEP.groupby(["idx", "method"]).agg(ARI=("ARI", "mean"), K=("n_clusters_found", "mean")).reset_index()
        dm["D"] = dm.idx.map(prof)
        for m, v in [("DipDECK", -0.98), ("IDEC", -0.59), ("DCN", -0.52)]:
            claim(f"deep r(D, ARI) {m}", v)(lambda m=m: pearsonr(dm[dm.method == m].D, dm[dm.method == m].ARI)[0])
        claim("deep rho(D, K) DipDECK", -0.93)(lambda: spearmanr(dm[dm.method == "DipDECK"].D, dm[dm.method == "DipDECK"].K).statistic)
        claim("deep DipDECK K hard", 1.4, 0.1)(lambda: dm[(dm.method == "DipDECK") & (dm.idx >= 50)].K.mean())

    # ---------------- Q4 cross-generator ----------------
    if XG is not None:
        def xg(gen, proto, col):
            d = XG[(XG.generator == gen) & (XG.protocol == proto)]
            agg = d.groupby(["native_param", "seed"]).agg(ARI=("ARI", "mean"), v=(col, "first")); return pearsonr(agg.v, agg.ARI)[0]
        claim("crossgen MDCGen F r(native, ARI)", -0.98)(lambda: xg("mdcgen", "fixed", "native_param"))
        claim("crossgen MDCGen F r(D, ARI)", -0.99)(lambda: xg("mdcgen", "fixed", "D_dataset"))
        claim("crossgen HAWKS F r(native, ARI)", 0.91)(lambda: xg("hawks", "fixed", "native_param"))
        claim("crossgen HAWKS F r(D, ARI)", -0.89)(lambda: xg("hawks", "fixed", "D_dataset"))
        claim("crossgen HAWKS S r(D_knn, ARI)", -0.93)(lambda: xg("hawks", "silhouette", "D_knn"))
        for gen, col, stated in [("mdcgen", "D_dataset", -0.95), ("mdcgen", "native_param", -0.95),
                                 ("hawks", "D_dataset", -0.92), ("hawks", "native_param", 0.90)]:
            claim(f"crossgen {gen} S r({col}, ARI)", stated, 0.005)(lambda gen=gen, col=col: xg(gen, "silhouette", col))
        claim("crossgen: runs without admissible configuration (DBSCAN, S)", 2, 0)(
            lambda: float((XG.get("run_status", pd.Series(dtype=str)) == "no_admissible_configuration").sum()))

    # ---------------- Q6 calibrated generation ----------------
    E6 = load(f"{C}/exp6_calibrated_generation/datasets.csv"); P6 = load(f"{C}/exp6_calibrated_generation/pool.csv")
    if E6 is not None:
        E6["err_imp"] = (E6.knn_edge_impurity - E6.target_edge_impurity).abs()
        claim("calibrated hit rate", 0.975, 0.001)(lambda: E6[E6.route == "calibrated"].hit.mean())
        claim("parametric hit rate", 0.725, 0.001)(lambda: E6[E6.route == "parametric"].hit.mean())
        claim("calibrated evals per dataset", 7.0, 0.5)(lambda: E6[E6.route == "calibrated"].n_evals.mean())
        claim("parametric spread of impurity", 0.019, 0.003)(lambda: E6[E6.route == "parametric"].groupby("target_id").knn_edge_impurity.std().mean())
        claim("parametric hit at target 0.2/0.2", 0.40, 0.01)(lambda: E6[(E6.route == "parametric") & (E6.target_edge_impurity == 0.2) & (E6.target_S_noise == 0.2)].hit.mean())
        if P6 is not None:
            sdA = lambda r: P6[P6.route == r].groupby(["target_id", "algorithm"]).ARI.std().mean()
            claim("ARI spread calibrated", 0.041, 0.005)(lambda: sdA("calibrated"))
            claim("ARI spread parametric", 0.035, 0.005)(lambda: sdA("parametric"))

    # ---------------- raw vs refined reference on the emblematic sweeps ----------------
    for name, ref, rawf in [("eta_int", "sweeps_v2/sweep_eta_int_fixed_eta0.3.csv", "sweeps_v2/sweep_eta_int_fixed_eta0.3_noprerefine.csv"),
                           ("imbalance", "sweeps_v2/sweep_imbalance_fixed.csv", "sweeps_v2/sweep_imbalance_fixed_noprerefine.csv")]:
        A, B = load(f"{C}/{ref}"), load(f"{C}/{rawf}")
        if A is not None and B is not None:
            common = sorted(set(A.algorithm) & set(B.algorithm))
            def maxdiff(A=A, B=B, common=common, col="ARI"):
                pa = A[A.algorithm.isin(common)].groupby(["algorithm", "value"])[col].mean(); pb = B[B.algorithm.isin(common)].groupby(["algorithm", "value"])[col].mean()
                return float((pa - pb).abs().max())
            claim(f"{name} sweep: raw vs refined reference, max |dARI|", 0.0, 0.005)(maxdiff)

    # ---------------- label refinement on the hard regime (k-NN, the default) ----------------
    # exp1 (protocol F) records in every row the fraction of labels changed by the refinement
    E1F = load(f"{C}/exp1_difficulty_validation/exp1_results_fixed.csv")
    if E1F is not None and "frac_relabelled" in E1F:
        FA = E1F.groupby(["difficulty", "config_id", "seed"]).frac_relabelled.first().reset_index()
        FR = FA[FA.difficulty == "hard"]
        claim("easy regime: labels changed by refinement", 0.0, 0.0005)(lambda: FA[FA.difficulty == "easy"].frac_relabelled.mean())
        claim("medium regime: labels changed by refinement", 0.001, 0.001)(lambda: FA[FA.difficulty == "medium"].frac_relabelled.mean())
        claim("hard regime: labels changed by k-NN refinement (mean)", 0.22, 0.01)(lambda: FR.frac_relabelled.mean())
        claim("hard regime: max over datasets", 0.37, 0.01)(lambda: FR.frac_relabelled.max())
        claim("hard regime: min variant mean", 0.15, 0.01)(lambda: FR.groupby("config_id").frac_relabelled.mean().min())
        claim("hard regime: max variant mean", 0.27, 0.01)(lambda: FR.groupby("config_id").frac_relabelled.mean().max())
    else:
        claim("refinement: frac_relabelled recorded in exp1_results_fixed.csv", 1.0, 0.0)(lambda: None)

    # ---------------- controls: refinement classifier (hard regime) and number of clusters ----------------
    E = f"{C}/exp1_difficulty_validation"
    RAW, SVM, RF = load(f"{E}/exp1_results_fixed_noprerefine.csv"), load(f"{E}/exp1_results_fixed_refinesvm_hard.csv"), load(f"{E}/exp1_results_fixed_refinerandom_forest_hard.csv")
    KNN = load(f"{E}/exp1_results_fixed.csv")
    if all(x is not None for x in (RAW, SVM, RF, KNN)):
        RAWh, KNNh = RAW[RAW.difficulty == "hard"], KNN[KNN.difficulty == "hard"]
        silent = lambda d, a: (((~d[d.algorithm == a].degenerate.astype(bool)) & (d[d.algorithm == a].ARI < 0.3)).mean())
        claim("refinement: labels changed by SVM (hard)", 0.04, 0.01)(lambda: SVM.groupby(["config_id", "seed"]).frac_relabelled.first().mean())
        claim("refinement: labels changed by RF (hard)", 0.0, 0.001)(lambda: RF.groupby(["config_id", "seed"]).frac_relabelled.first().mean())
        claim("refinement: Ward silent raw", 0.68, 0.01)(lambda: silent(RAWh, "Ward"))
        claim("refinement: Ward silent kNN", 0.92, 0.01)(lambda: silent(KNNh, "Ward"))
        claim("refinement: KMeans silent raw", 0.08, 0.01)(lambda: silent(RAWh, "KMeans"))
        claim("refinement: KMeans silent kNN", 0.24, 0.01)(lambda: silent(KNNh, "KMeans"))
        def maxdiff():
            algs = [a for a in ("KMeans", "GMM", "Spectral", "Ward", "Average", "Complete", "DBSCAN", "HDBSCAN", "OPTICS") ]
            m = pd.DataFrame({k: d[d.algorithm.isin(algs)].groupby("algorithm").ARI.mean() for k, d in (("raw", RAWh), ("knn", KNNh), ("svm", SVM), ("rf", RF))})
            return float((m.max(axis=1) - m.min(axis=1)).max())
        claim("refinement: max spread of mean ARI across references", 0.08, 0.01)(maxdiff)
    S = f"{C}/sweeps_v2"
    for K in (3, 8):
        EK = load(f"{S}/sweep_eta_int_fixed_eta0.3_K{K}.csv")
        if EK is not None:
            ends = EK.groupby(["algorithm", "value"]).ARI.mean()
            claim(f"K={K}: min bridge-end ARI of KMeans/GMM/Ward/Average", {3: 0.97, 8: 0.91}[K], 0.01)(lambda ends=ends: min(ends[(a, 1.0)] for a in ("KMeans", "GMM", "Ward", "Average")))
            claim(f"K={K}: min distant-end ARI of HDBSCAN/OPTICS", 0.99, 0.01)(lambda ends=ends: min(ends[(a, 0.0)] for a in ("HDBSCAN", "OPTICS")))

    # ---------------- predictive value of the envelope (written by audit_value_v2.py) ----------------
    PS = load("benchmarks/cross-generator/results/predictive_summary.csv")
    if PS is not None:
        v = PS.set_index(PS.columns[0])["value"]
        # values as printed in the text (two decimals; tolerance half a unit of the last digit, so the
        # text must be the correctly rounded value), with their 95% bootstrap intervals
        for key, stated in [("mae_env", 0.26), ("mae_bench", 0.39), ("mae_sil", 0.34),
                            ("gain_vs_bench", 0.13), ("gain_vs_bench_lo", 0.10), ("gain_vs_bench_hi", 0.17),
                            ("gain_vs_sil", 0.08), ("gain_vs_sil_lo", 0.04), ("gain_vs_sil_hi", 0.11),
                            ("mdcgen_gain_vs_bench", 0.09), ("mdcgen_gain_vs_bench_lo", 0.07), ("mdcgen_gain_vs_bench_hi", 0.11),
                            ("hawks_gain_vs_bench", 0.18), ("hawks_gain_vs_bench_lo", 0.12), ("hawks_gain_vs_bench_hi", 0.24),
                            ("hawks_gain_vs_sil", 0.17), ("hawks_gain_vs_sil_lo", 0.12), ("hawks_gain_vs_sil_hi", 0.22),
                            ("mdcgen_gain_vs_sil", -0.02), ("mdcgen_gain_vs_sil_lo", -0.05),
                            ("close_pairs_env", 0.54), ("close_pairs_sil", 0.69), ("close_pairs_bench", 0.54)]:
            claim(f"predictive: {key}", stated, 0.005)(lambda key=key: v.get(key))
        claim("predictive: mdcgen_gain_vs_sil_hi", 0.005, 0.0005)(lambda: v.get("mdcgen_gain_vs_sil_hi"))
        claim("predictive: gain vs leaderboard CI excludes 0", 1.0, 0.0)(lambda: float(v["gain_vs_bench_lo"] > 0))
        claim("predictive: gain vs Silhouette CI excludes 0", 1.0, 0.0)(lambda: float(v["gain_vs_sil_lo"] > 0))
        claim("predictive: MDCGen gain vs Silhouette CI includes 0", 1.0, 0.0)(lambda: float(v["mdcgen_gain_vs_sil_lo"] < 0 < v["mdcgen_gain_vs_sil_hi"]))
    if E6 is not None:
        claim("calibrated hits (of 40)", 39, 0)(lambda: E6[E6.route == "calibrated"].hit.sum())
        claim("parametric hits (of 40)", 29, 0)(lambda: E6[E6.route == "parametric"].hit.sum())

    SP = load(f"{C}/sweeps_v2/spectral_imbalance_check.csv")
    if SP is not None:
        claim("Spectral rho=5: mean F1 (all points) reproduces Table 8", 0.50, 0.01)(lambda: SP.f1_all_points.mean())
        claim("Spectral rho=5: mean inlier ARI", 1.00, 0.005)(lambda: SP.inlier_ARI.mean())
        claim("Spectral rho=5: seeds with small cluster intact", 4, 0)(lambda: int((SP.of_small == SP.small_size).sum() - (SP.other_inliers > 0).sum()))
        claim("Spectral rho=5: min noise attached", 4, 0)(lambda: SP.true_noise.min())
        claim("Spectral rho=5: max noise attached", 33, 0)(lambda: SP.true_noise.max())

    # ---------------- pooled set of Section 6.1 (computed by make_paper_tables_v2.pooled_summary) ----------------
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_mpt", os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_paper_tables_v2.py"))
    _mpt = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mpt)
    PSUM = _mpt.pooled_summary(R)
    if PSUM is not None:
        claim("pooled: N datasets", 450, 0)(lambda: PSUM["N_pooled"])
        for key, stated in [("r_pooled_D", -0.75), ("r_pooled_D_knn", -0.76), ("r_pooled_one_minus_silhouette", -0.65),
                            ("r_pooled_knn_cv_error", -0.51), ("r_noise_one_minus_silhouette", -0.14),
                            ("R2_k10_components_knn", 0.73), ("R2_k10_baselines", 0.41)]:
            claim(f"pooled: {key}", stated, 0.005)(lambda key=key: PSUM[key])

    # ---------------- reporting conventions: ordering of critical levels across cut-offs ----------------
    def _crit(df, thr):
        m = df.groupby(["algorithm", "value"]).ARI.mean().unstack()
        return {a: (row[row < thr].index.min() if (row < thr).any() else 99.0) for a, row in m.iterrows()}
    def _min_rho():
        out = []
        for df in (OVF, ANIF):
            c = {t: _crit(df, t) for t in (0.4, 0.5, 0.6)}; algs = sorted(c[0.5])
            for a_, b_ in ((0.4, 0.5), (0.5, 0.6)):
                out.append(spearmanr([c[a_][x] for x in algs], [c[b_][x] for x in algs]).statistic)
        return min(out)
    if OVF is not None and ANIF is not None:
        claim("conventions: min rank agreement of critical levels", 0.87, 0.01)(_min_rho)

    # ---------------- paired seeds (S6.7) ----------------
    PR = load(f"{C}/sweeps_v2/paired_ratios.csv")
    if PR is not None:
        claim("paired: method-sweep pairs", 40, 0)(lambda: len(PR))
        claim("paired: pairs with SE ratio <= 0.90", 23, 0)(lambda: int((PR.ratio <= 0.90).sum()))
        claim("paired: Ward inter-density ratio", 0.21, 0.01)(lambda: float(PR[(PR.algorithm == "Ward") & PR.sweep.str.startswith("inter")].ratio.iloc[0]))
    else:
        claim("paired: method-sweep pairs", 40, 0)(lambda: None)

    # ---------------- intra-density (core-dense) sweep, protocol F (Section 6.2) ----------------
    INTRA = load(f"{C}/sweeps_v2/sweep_intra_density_fixed.csv")
    if INTRA is not None:
        mi = INTRA.groupby(["algorithm", "value"]).ARI.mean().unstack(); lo, hi = mi.columns.min(), mi.columns.max()
        for alg, a0, a1 in [("DBSCAN", 0.78, 0.92), ("HDBSCAN", 0.58, 0.80)]:
            claim(f"intra-density: {alg} ARI at lowest level", a0, 0.005)(lambda alg=alg: mi.loc[alg, lo])
            claim(f"intra-density: {alg} ARI at highest level", a1, 0.005)(lambda alg=alg: mi.loc[alg, hi])
        claim("intra-density: no method lower at the highest level by more than 0.02", 1.0, 0.0)(lambda: float((mi[hi] >= mi[lo] - 0.025).all()))
        claim("intra-density: fragmentation at highest level", 0.49, 0.005)(lambda: INTRA[INTRA.value == hi].knn_fragmentation.mean())
    else:
        claim("intra-density: sweep present", 1.0, 0.0)(lambda: None)

    # ---------------- size of the operating envelope (training set of the predictor, Section 6.4) ----------------
    def _envelope_sizes():
        def wd(a, b):
            if a is None: return b
            return pd.concat([a, b]) if (b is not None and "DPC" not in set(a.algorithm)) else a
        parts = [wd(load(f"{C}/exp1_difficulty_validation/exp1_results_fixed.csv"), load(f"{C}/exp1_difficulty_validation/exp1_results_fixed_DPC.csv"))]
        parts += [wd(load(f"{C}/sweeps_v2/sweep_{k}_fixed.csv"), load(f"{C}/sweeps_v2/sweep_{k}_fixed_DPC.csv"))
                  for k in ["overlap", "imbalance", "inter_density", "intra_density", "connectivity"]]
        parts += [load(f"{C}/sweeps_v2/sweep_eta_int_fixed_eta0.3.csv"), load(f"{C}/exp2_sensitivity_analysis/exp2_results_fixed_anisotropy.csv")]
        d = pd.concat([p_ for p_ in parts if p_ is not None]).dropna(subset=["ARI"])
        return d.groupby("algorithm").size()
    claim("envelope: datasets per method (eight classical methods)", 250, 0)(lambda: float(_envelope_sizes().drop(["DPC", "Complete"], errors="ignore").unique().item()))
    claim("envelope: datasets for DPC", 230, 0)(lambda: float(_envelope_sizes()["DPC"]))

    # ---------------- what the noise interventions move (Section 6.3, factor x descriptor matrix) ----------------
    if NOISE is not None:
        DESC = ["S_separation", "S_boundary", "S_shape", "knn_edge_impurity", "knn_ambiguous_fraction", "knn_fragmentation",
                "knn_density_heterogeneity", "knn_intra_density_spread", "knn_intrinsic_dim_twonn", "base_one_minus_silhouette"]
        def _rho(df, key, col):
            d = df.groupby([key, "seed"])[col].first().reset_index()
            return 0.0 if d[col].isna().all() or d[col].nunique(dropna=True) <= 1 else spearmanr(d[key], d[col], nan_policy="omit").statistic
        N3 = NOISE[NOISE.value_eta == 0.3]
        claim("eta_int at eta=0.3: max |rho| over the other descriptors", 0.0, 0.01)(lambda: max(abs(_rho(N3, "value_eta_int", c)) for c in DESC))
        claim("eta_int at eta=0.3: rho with noise proximity", 0.91, 0.005)(lambda: _rho(N3, "value_eta_int", "knn_noise_proximity"))
        for col, stated_rho in [("knn_intrinsic_dim_twonn", -0.40), ("knn_fragmentation", -0.37), ("knn_noise_proximity", 0.46), ("S_shape", 0.43)]:
            claim(f"eta: rho with {col}", stated_rho, 0.005)(lambda col=col: _rho(NOISE, "value_eta", col))

    # ---------------- instrument sensitivity (Section 6.2, Table S5) ----------------
    Q2 = load("analysis/tables/q2_instrument_sensitivity.csv")
    if Q2 is not None:
        Q2i = Q2.set_index("perturbation")
        claim("instrument: min ordering rho of D", 0.984, 0.0005)(lambda: Q2["ordering_rho[D]"].min())
        claim("instrument: min ordering rho of S_sep and S_bnd (> 0.94)", 0.945, 0.005)(lambda: Q2[["ordering_rho[S_separation]", "ordering_rho[S_boundary]"]].min().min())
        claim("instrument: S_shape under random forest", 0.47, 0.005)(lambda: Q2i.loc["random forest", "ordering_rho[S_shape]"])
        claim("instrument: S_shape with k = 2 sqrt(n)", 0.71, 0.005)(lambda: Q2i.loc["knn k=2sqrt(n)", "ordering_rho[S_shape]"])
        claim("instrument: k-NN descriptors unchanged (min rho)", 1.0, 0.0005)(lambda: Q2[["ordering_rho[knn_edge_impurity]", "ordering_rho[knn_fragmentation]", "ordering_rho[knn_noise_proximity]"]].min().min())
        claim("instrument: S_noise min rho over q_gap", 0.995, 0.0005)(lambda: Q2i.loc[["q_gap=0.05", "q_gap=0.25"], "ordering_rho[S_noise]"].min())
        claim("instrument: Q1 correlation, default", -0.92, 0.005)(lambda: Q2i.loc["default", "r(D, meanARI)"])
        claim("instrument: Q1 correlation, min", -0.95, 0.005)(lambda: Q2["r(D, meanARI)"].min())
        claim("instrument: Q1 correlation, max", -0.90, 0.005)(lambda: Q2["r(D, meanARI)"].max())
    else:
        claim("instrument: q2 table present", 1.0, 0.0)(lambda: None)

    # ---------------- every run recorded: tunings without admissible configuration, errors (Supplement S4) ----------------
    def _na(path, extra=None):
        d = load(path)
        if d is None or "run_status" not in d: return None
        d = d[d.run_status == "no_admissible_configuration"]
        if extra is not None: d = d[extra(d)]
        return d.algorithm.value_counts().to_dict()
    for path, expect, extra in [
            (f"{C}/exp1_difficulty_validation/exp1_results.csv", {"DBSCAN": 21, "HDBSCAN": 2, "OPTICS": 1}, None),
            (f"{C}/exp1_difficulty_validation/exp1_results_dbcv.csv", {"DBSCAN": 21, "HDBSCAN": 2, "OPTICS": 1}, None),
            (f"{C}/exp2_sensitivity_analysis/exp2_results_silhouette_noise.csv", {"DBSCAN": 86}, None),
            (f"{C}/exp2_sensitivity_analysis/exp2_results.csv", {"DBSCAN": 17, "OPTICS": 6, "HDBSCAN": 3}, lambda d: d.parameter == "anisotropy"),
            (f"{C}/sweeps_v2/sweep_overlap_silhouette.csv", {"DBSCAN": 10, "OPTICS": 3, "HDBSCAN": 2}, None)]:
        claim(f"no admissible configuration: {os.path.basename(path)}", 1.0, 0.0)(lambda path=path, expect=expect, extra=extra: float(_na(path, extra) == expect))
    def _total_errors_and_others():
        n_err, n_other = 0, 0
        for f in glob.glob(os.path.join(R, "benchmarks", "**", "*.csv"), recursive=True):
            d = pd.read_csv(f, float_precision="round_trip")
            if "failure_mode" in d: n_err += int((d.failure_mode == "error").sum())
            if "run_status" in d:   # only the five files above and the cross-generator file have such runs
                if os.path.basename(f) not in ("exp1_results.csv", "exp1_results_dbcv.csv", "exp2_results_silhouette_noise.csv",
                                               "exp2_results.csv", "sweep_overlap_silhouette.csv", "crossgen_results.csv"):
                    n_other += int((d.run_status == "no_admissible_configuration").sum())
        return n_err + n_other
    claim("no error row, no other run without admissible configuration", 0, 0)(_total_errors_and_others)
    claim("noise grid: DBSCAN runs without admissible configuration all have eta >= 0.15 and eta_int <= 0.75", 1.0, 0.0)(
        lambda: float(NOISE[NOISE.run_status == "no_admissible_configuration"].pipe(lambda d: ((d.value_eta_int <= 0.75) & (d.value_eta >= 0.15)).all())))

    # ---------------- ablation and grouped cross-validation, protocol F (Section 6.1, Tables 4 and S...) ----------------
    if E1F is not None:
        _F = E1F[E1F.algorithm != "DPC"].copy(); _F["family"] = _F.algorithm.map(_mpt.FAMILY)
        SETS = dict(_mpt.SETS)
        def _abl(group, set_name):
            g = _F if group == "All" else _F[_F.family == group]
            agg = _mpt.ds_level(g); cols = [c for c in SETS[set_name] if c in agg]
            return _mpt.loo_r2(agg[cols], agg.ARI)
        SIL, DSC, COMP4, KNNs, CK = ("$1-$Silhouette (reference labels)", "$D$ (fixed-weight scalar)", "four components, separately",
                                      "$k$-NN descriptors", "components $+$ $k$-NN descriptors")
        for group, set_name, stated in [("All", SIL, 0.87), ("All", COMP4, 0.92), ("All", DSC, 0.84), ("connectivity", SIL, 0.32),
                                        ("connectivity", CK, 0.84), ("graph", SIL, 0.28), ("graph", CK, 0.78), ("density", SIL, 0.93),
                                        ("All", "$S_{\\text{shape}}$ alone", 0.12)]:
            claim(f"ablation F, {group}: {set_name[:28]}", stated, 0.005)(lambda group=group, set_name=set_name: _abl(group, set_name))
        _agg = _mpt.ds_level(E1F[E1F.algorithm != "DPC"]); _reg = _agg.config_id // 5
        def _loro(cols): return _mpt.loo_r2(_agg[cols], _agg.ARI, "g", _reg)
        claim("LORO: full profile", 0.62, 0.005)(lambda: _loro(_mpt.COMP + _mpt.KNN + _mpt.BASE))
        claim("LOO: full profile", 0.97, 0.005)(lambda: _mpt.loo_r2(_agg[_mpt.COMP + _mpt.KNN + _mpt.BASE], _agg.ARI))
        claim("LORO: components + k-NN", 0.45, 0.005)(lambda: _loro(_mpt.COMP + _mpt.KNN))
        claim("LORO: Silhouette", 0.87, 0.005)(lambda: _loro(["base_one_minus_silhouette"]))
        claim("LORO: k-NN descriptors negative", 1.0, 0.0)(lambda: float(_loro(_mpt.KNN) < 0))

    # ---------------- run ----------------
    # Every claim must be evaluated: a block whose input file is missing registers fewer claims, and
    # that is a failure, not a smaller denominator. Update EXPECTED_CLAIMS when claims are added.
    EXPECTED_CLAIMS = 243
    fails = 0; print(f"{'claim':52s} {'paper':>8s} {'data':>8s}  status")
    if len(CLAIMS) != EXPECTED_CLAIMS:
        print(f"only {len(CLAIMS)} of the {EXPECTED_CLAIMS} claims could be set up: input result files are missing")
        fails += EXPECTED_CLAIMS - len(CLAIMS)
    for label, stated, fn, tol in CLAIMS:
        try:
            got = float(fn())
        except Exception as e:
            print(f"{label:52s} {stated:8} {'n/a':>8s}  NOT COMPUTED ({type(e).__name__}: {e})"); fails += 1; continue
        ok = (np.isinf(stated) and np.isinf(got)) or abs(got - stated) <= tol
        fails += (not ok)
        print(f"{label:52s} {stated:8.3f} {got:8.3f}  {'ok' if ok else 'MISMATCH'}")
    print(f"\n{EXPECTED_CLAIMS - fails}/{EXPECTED_CLAIMS} claims match (tolerance per claim, default ±0.015)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
