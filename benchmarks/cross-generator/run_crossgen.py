#!/usr/bin/env python
"""
Cross-generator confrontation: MDCGen and HAWKS (paper Q4).

For each external generator, sweep its native difficulty axis (7 levels x 5 seeds),
compute the difficulty profile of every dataset with the *same* instrument as for
the in-house suite, run the classical pool under protocols F and S, and write one
row per (dataset, algorithm, protocol) with the native parameter, the profile and
the full evaluation. The analysis (native parameter vs profile as predictors of
ARI, per family) is `analyze_crossgen.py`.

Requirements (not part of the core package):
    pip install deap tqdm hawks --no-deps      # hawks 0.2.0
    pip install git+https://github.com/CN-TU/mdcgenpy
hawks 0.2.0 needs two compatibility shims (pandas >= 2 removed DataFrame.append;
Orange is imported for a plot we never draw); they are applied below.

    python benchmarks/cross-generator/run_crossgen.py [--seeds 42 123 ...] [--gens 50]
"""
from __future__ import annotations
import argparse, os, sys, types, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# --- compatibility shims for hawks 0.2.0 -------------------------------------
_o = types.ModuleType("Orange"); _o.evaluation = types.ModuleType("Orange.evaluation")
sys.modules.setdefault("Orange", _o); sys.modules.setdefault("Orange.evaluation", _o.evaluation)
if not hasattr(pd.DataFrame, "append"):
    def _append(self, other, ignore_index=False, **kw):
        other = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other] if isinstance(other, dict) else other)
        return pd.concat([self, other], ignore_index=ignore_index)
    pd.DataFrame.append = _append

from clustaudit.metrics import difficulty_profile
from clustaudit.audit import run_pool, evaluate_partition, evaluate_run, result_row, ALGORITHMS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
N, D, K = 1000, 20, 5
POOL = ["KMeans", "GMM", "Spectral", "Ward", "Average", "DBSCAN", "HDBSCAN", "OPTICS", "DPC"]
# native axes (calibrated so that the reference Silhouette spans ~0.9 -> ~0.15)
MDC_COMPACTNESS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
HAWKS_TARGET_SIL = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def gen_mdcgen(level, seed):
    from mdcgenpy.clusters import ClusterGenerator
    cg = ClusterGenerator(seed=seed, n_samples=N, n_feats=D, k=K, compactness_factor=level, outliers=0)
    X, y = cg.generate_data()
    return np.asarray(X, float), np.ravel(np.asarray(y)).astype(int)


def gen_hawks(level, seed, gens):
    import hawks
    cfg = {"hawks": {"seed_num": seed, "num_runs": 1},
           "dataset": {"num_examples": N, "num_clusters": K, "num_dims": D},
           "objectives": {"silhouette": {"target": level}},
           "ga": {"num_gens": gens}}
    g = hawks.create_generator(cfg); g.run()
    datasets, labels = g.get_best_dataset()
    return np.asarray(datasets[0], float), np.asarray(labels[0]).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456, 789, 1011])
    ap.add_argument("--gens", type=int, default=50, help="HAWKS generations")
    ap.add_argument("--generators", nargs="*", default=["mdcgen", "hawks"])
    ap.add_argument("--protocols", nargs="*", default=["fixed", "silhouette"])
    a = ap.parse_args()
    rows = []
    for gname in a.generators:
        levels = MDC_COMPACTNESS if gname == "mdcgen" else HAWKS_TARGET_SIL
        for lv in levels:
            for seed in a.seeds:
                t0 = time.time()
                X, y = gen_mdcgen(lv, seed) if gname == "mdcgen" else gen_hawks(lv, seed, a.gens)
                prof = difficulty_profile(X, y, seed=seed).to_dict()
                from sklearn.metrics import silhouette_score
                sil_ref = silhouette_score(X, y)
                for proto in a.protocols:
                    out = run_pool(X, seed, POOL, proto, n_clusters=K, with_status=True)   # no run dropped
                    for alg, (lab, status) in out.items():
                        rows.append({"generator": gname, "native_param": lv, "seed": seed, "protocol": proto,
                                     "algorithm": alg, "silhouette_ref": sil_ref,
                                     **{k: v for k, v in prof.items() if k not in ("n_samples", "dimension", "n_clusters")},
                                     **evaluate_run(X, y, lab, status)})
                print(f"[{gname} {lv} seed={seed}] sil_ref={sil_ref:.3f} D={prof['D_dataset']:.3f} ({time.time()-t0:.0f}s)")
                pd.DataFrame(rows).to_csv(os.path.join(OUT, "crossgen_results.csv"), index=False)
    print(f"-> {OUT}/crossgen_results.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
