#!/usr/bin/env python
"""
Exp 6 -- Calibrated generation: control the difficulty a dataset *has*, not the parameter it was given.

For a grid of targets on two measured descriptors (edge impurity of the labelled k-NN graph,
S_noise) and a set of seeds, two ways of producing a dataset are compared at equal budget:

  parametric : the generator parameters are set once (calibrated on a reference seed, so that the
               parametric route is given the best single parameter choice it can have) and reused
               for every other seed. This is what a sweep does.
  calibrated : the parameters are searched for every seed until the *measured* descriptors are
               within tolerance of the target (clustaudit.generators.calibrate).

Reported, per target: hit rate within tolerance, mean absolute error on each descriptor, the spread
of the realized descriptors over seeds, and, after running the fixed-protocol pool on both sets of
datasets, the spread of each method's ARI over seeds at fixed target (does controlling the measured
difficulty remove unexplained outcome variance?).

    python benchmarks/classical/exp6_calibrated_generation.py [--smoke] [--budget 40] [--no-pool]

Output: results/exp6_calibrated_generation/{datasets,pool}.csv
"""
from __future__ import annotations
import argparse, os, sys, time, itertools
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); sys.path.insert(0, ROOT)
from clustaudit.generators.calibrate import calibrate, _measure, operating_point
from clustaudit.audit.algorithms import run_pool, CLASSICAL_EIGHT
from clustaudit.audit.evaluation import evaluate_partition, result_row, evaluate_run

OUT = os.path.join(HERE, "results", "exp6_calibrated_generation")
TARGETS = {"knn_edge_impurity": [0.05, 0.10, 0.20, 0.30], "S_noise": [0.20, 0.50]}
TOL = {"knn_edge_impurity": 0.03, "S_noise": 0.05}
REF_SEED = 42
SEEDS = [123, 456, 789, 1011, 2024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--no-pool", action="store_true"); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--d", type=int, default=20)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    targets = list(itertools.product(TARGETS["knn_edge_impurity"], TARGETS["S_noise"]))
    seeds = SEEDS[:2] if a.smoke else SEEDS
    if a.smoke: targets = targets[:2]
    pool = CLASSICAL_EIGHT + ["DPC"]
    rows, prow = [], []
    t0 = time.time()
    for ti, (imp, sn) in enumerate(targets):
        target = {"knn_edge_impurity": imp, "S_noise": sn}
        # the parametric route gets the best single parameter choice: calibrated once on the reference seed
        ref = calibrate(target, seed=REF_SEED, n=a.n, d=a.d, tolerance=TOL, budget=a.budget, keep_state=False)
        print(f"[{ti+1}/{len(targets)}] target {target}: reference params {ref.params} (hit={ref.hit}, {ref.n_evals} evals)")
        for seed in seeds:
            datasets = {}
            st, pr = _measure(ref.params, a.n, a.d, 5, seed, operating_point()); datasets["parametric"] = (st, pr, ref.params, 1, None)
            cal = calibrate(target, seed=seed, n=a.n, d=a.d, tolerance=TOL, budget=a.budget, start=ref.params)
            datasets["calibrated"] = (cal.state, cal.profile, cal.params, cal.n_evals, cal.hit)
            for route, (state, prof, params, nev, hit) in datasets.items():
                hit = all(abs(prof[k] - target[k]) <= TOL[k] for k in target) if hit is None else hit
                rows.append({"route": route, "target_id": ti, "target_edge_impurity": imp, "target_S_noise": sn, "seed": seed, "n_evals": nev, "hit": hit,
                             **{f"param_{k}": v for k, v in params.items()}, **{k: prof[k] for k in prof if isinstance(prof[k], (int, float, np.floating))}})
                if not a.no_pool:
                    labels = run_pool(state.Z, seed, pool, "fixed", n_clusters=5, with_status=True)   # no run dropped
                    for name, (lab, status) in labels.items():
                        prow.append({"route": route, "target_id": ti, "target_edge_impurity": imp, "target_S_noise": sn, "seed": seed, "algorithm": name,
                                     **evaluate_run(state.Z, state.y, lab, status)})
            print(f"   seed {seed}: parametric imp={datasets['parametric'][1]['knn_edge_impurity']:.3f} Sn={datasets['parametric'][1]['S_noise']:.3f} | "
                  f"calibrated imp={cal.profile['knn_edge_impurity']:.3f} Sn={cal.profile['S_noise']:.3f} hit={cal.hit} ({cal.n_evals} evals)  [{time.time()-t0:.0f}s]")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "datasets.csv"), index=False)
        if prow: pd.DataFrame(prow).to_csv(os.path.join(OUT, "pool.csv"), index=False)
    df = pd.DataFrame(rows)
    df["err_imp"] = (df.knn_edge_impurity - df.target_edge_impurity).abs(); df["err_sn"] = (df.S_noise - df.target_S_noise).abs()
    print("\nhit rate / MAE per route:"); print(df.groupby("route")[["hit", "err_imp", "err_sn", "n_evals"]].mean().round(3))
    print("\nspread of the realized descriptors over seeds (mean over targets of the std):")
    print(df.groupby(["route", "target_id"])[["knn_edge_impurity", "S_noise"]].std().groupby("route").mean().round(3))
    if prow:
        p = pd.DataFrame(prow); sd = p.groupby(["route", "target_id", "algorithm"]).ARI.std().groupby(["route", "algorithm"]).mean().unstack("route").round(3)
        print("\nstd of ARI over seeds at fixed target, per method:"); print(sd)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
