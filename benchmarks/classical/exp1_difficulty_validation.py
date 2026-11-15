#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 1: D_dataset Validation
===================================

Demonstrates that intrinsic difficulty score D_dataset correlates with 
observed algorithmic performance degradation.

Expected Result: Strong negative correlation (r < -0.7) between D_dataset 
and average ARI across algorithms.

Author: R. Riad
Date: March 2026
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import warnings
from scipy.stats import spearmanr, pearsonr

# --- imports: `clustaudit` must be installed (pip install -e .); fallback to repo root ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
try:
    import clustaudit  # noqa: F401
except ImportError:
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_HERE, _os.pardir, _os.pardir)))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
# --- end imports ---

from clustaudit.core.pipeline import generate_dataset
from clustaudit.config.params import (
    StructuralParams,
    NoiseParams,
    AmbiguityParams,
    DensityParams,
)
from clustaudit.metrics import compute_metrics_with_and_without_noise, profile_from_state

from tune_algo import (
    result_row,
    evaluate_run,
    error_row,
    tuning_status,
    tune_dpc,
    tune_kmeans,
    tune_agglomerative,
    tune_spectral,
    tune_gmm,
    tune_dbscan,
    tune_hdbscan,
    tune_optics,
    get_labels,
    compute_metrics,
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

SEEDS = [42, 123, 456, 789, 1011]
ALGOS = None   # set by --algos
NO_PREREFINE = False
REFINE_CLF = None
REGIME = None
import tune_algo as _tune_algo
OUTPUT_DIR = Path(_HERE) / "results" / "exp1_difficulty_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# =============================================================================
# CONFIGURATION GENERATOR
# =============================================================================

# The 15 configurations are defined once, in clustaudit.io.suite (shared with
# `clustaudit suite`, so the classical and deep tiers audit the same corpus).
from clustaudit.io.suite import create_difficulty_configs  # noqa: E402


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_experiment():
    """
    Experiment 1: D_dataset Validation
    
    Objective: Demonstrate correlation between D_dataset and performance
    Expected: Strong negative correlation (r < -0.7) between D_dataset and average ARI
    """
    print("="*80)
    print("EXPERIMENT 1: D_DATASET VALIDATION")
    print("="*80)
    
    configs = create_difficulty_configs()
    results = []
    
    total_jobs = len(configs) * len(SEEDS)
    current_job = 0
    
    for config in configs:
        for seed in SEEDS:
            current_job += 1
            print(f"[{current_job}/{total_jobs}] Config {config['config_id']} "
                  f"({config['difficulty']}, var={config['variant']}), seed={seed}")
            
            try:
                amb = copy.deepcopy(config['ambiguity'])
                if NO_PREREFINE:
                    amb.enable_pre_refinement = False
                if REFINE_CLF:
                    amb.pre_refinement_classifier = REFINE_CLF
                state = generate_dataset(
                    n_samples=1000,
                    dimension=20,
                    n_clusters=5,
                    structural=config['structural'],
                    noise=config['noise'],
                    ambiguity=amb,
                    random_seed=seed,
                )
                
                X, y_true = state.Z, state.y
                prof = profile_from_state(state, seed=seed)
                complexity = prof.score
                prof_cols = {k: v for k, v in prof.to_dict().items()
                             if k not in ('D_dataset', 'n_samples', 'dimension', 'n_clusters')}
                
                algorithms = {
                    "KMeans": tune_kmeans(X, seed),
                    "DBSCAN": tune_dbscan(X),
                    "HDBSCAN": tune_hdbscan(X),
                    "OPTICS": tune_optics(X),
                    "Ward": tune_agglomerative(X, "ward"),
                    "Average": tune_agglomerative(X, "average"),
                    "Complete": tune_agglomerative(X, "complete"),
                    "Spectral": tune_spectral(X, seed),
                    "GMM": tune_gmm(X, seed),
                    "DPC": tune_dpc(X),
                }
                if ALGOS:
                    algorithms = {k: v for k, v in algorithms.items() if k in ALGOS}
                
                for algo_name, algo in algorithms.items():
                    # every run is recorded: a tuning without an admissible configuration gives a
                    # degenerate output (detectable failure), a method that raises gives an error row
                    try:
                        labels = get_labels(algo, X)
                        res = evaluate_run(X, y_true, labels, tuning_status(algo))
                    except Exception as e:
                        print(f"    ! {algo_name} failed: {type(e).__name__}: {e} (recorded as an error row)")
                        res = error_row(f"{type(e).__name__}: {e}")
                    results.append({
                        'config_id': config['config_id'],
                        'difficulty': config['difficulty'],
                        'variant': config['variant'],
                        'seed': seed,
                        'D_dataset': complexity,
                        'frac_relabelled': state.metadata.get('frac_relabelled', 0.0),
                        **prof_cols,
                        'algorithm': algo_name,
                        **res,
                    })
                
            except Exception as e:
                print(f"  ✗ Dataset generation failed: {str(e)}")
                continue
    
    df_results = pd.DataFrame(results)
    df_results["criterion"] = _tune_algo.CRITERION
    out_csv = OUTPUT_DIR / ("exp1_results.csv" if _tune_algo.CRITERION == "silhouette"
                            else f"exp1_results_{_tune_algo.CRITERION}.csv")
    if ALGOS:
        out_csv = out_csv.with_name(out_csv.stem + "_" + "_".join(ALGOS) + ".csv")
    if NO_PREREFINE:
        out_csv = out_csv.with_name(out_csv.stem + "_noprerefine.csv")
    if REFINE_CLF:
        out_csv = out_csv.with_name(out_csv.stem + f"_refine{REFINE_CLF}.csv")
    if REGIME:
        out_csv = out_csv.with_name(out_csv.stem + f"_{REGIME}.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\n✓ Results saved: {out_csv}")
    
    return df_results


# =============================================================================
# PLOTTING & ANALYSIS
# =============================================================================

def plot_results(df_results: pd.DataFrame):
    """Create Figure 4: D_dataset vs Performance"""
    print("\nGenerating plots...")
    
    # Aggregate across seeds
    df_agg = df_results.groupby(['D_dataset', 'algorithm']).agg({
        'ARI': ['mean', 'std'],
        'NMI': ['mean', 'std'],
        'Silhouette': ['mean', 'std'],
        'DB_Index': ['mean', 'std'],
        'CH_Index': ['mean', 'std'],
    }).reset_index()
    
    for metric in ['ARI', 'NMI', 'Silhouette', 'DB_Index', 'CH_Index']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for algo in sorted(df_results['algorithm'].unique()):
            data = df_agg[df_agg['algorithm'] == algo]
            if len(data) == 0:
                continue
            
            mean = data[(metric, 'mean')].values
            std = data[(metric, 'std')].values
            d_vals = data['D_dataset'].values
            
            ax.plot(d_vals, mean, marker='o', label=algo, linewidth=2)
            ax.fill_between(d_vals, mean - std, mean + std, alpha=0.2)
        
        ax.set_xlabel('Dataset Difficulty Score (D_dataset)', fontsize=12)
        ax.set_ylabel(f'{metric}', fontsize=12)
        ax.set_title(f'Algorithm Performance vs Intrinsic Difficulty ({metric})', fontsize=14)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"figure_difficulty_vs_{metric.lower()}.pdf", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: figure_difficulty_vs_{metric.lower()}.pdf")


def print_summary(df_results: pd.DataFrame):
    """Print statistical summary with correlation analysis"""
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY & CORRELATION ANALYSIS")
    print("="*80)
    
    df_avg = df_results.groupby('D_dataset')[['ARI', 'NMI', 'Silhouette', 'DB_Index', 'CH_Index']].mean()
    
    print("\nAverage metrics by difficulty:")
    print(df_avg.round(3))
    
    print("\nPer-algorithm performance:")
    df_algo = df_results.groupby('algorithm')[['ARI', 'NMI', 'Silhouette', 'DB_Index', 'CH_Index']].mean()
    print(df_algo.round(3))
    
    # Statistical correlation analysis
    print("\n" + "-"*80)
    print("CORRELATION ANALYSIS: D_dataset vs Performance Metrics")
    print("-"*80)
    
    # Compute mean metrics across algorithms for each D_dataset value
    df_by_difficulty = df_results.groupby('D_dataset')[['ARI', 'NMI', 'Silhouette']].mean()
    
    for metric in ['ARI', 'NMI', 'Silhouette']:
        d_vals = df_by_difficulty.index.values
        metric_vals = df_by_difficulty[metric].values
        
        # Remove NaN values for correlation
        valid_idx = ~(np.isnan(d_vals) | np.isnan(metric_vals))
        d_vals_clean = d_vals[valid_idx]
        metric_vals_clean = metric_vals[valid_idx]
        
        if len(d_vals_clean) > 2:
            r_pearson, p_pearson = pearsonr(d_vals_clean, metric_vals_clean)
            r_spearman, p_spearman = spearmanr(d_vals_clean, metric_vals_clean)
            
            print(f"\n{metric}:")
            print(f"  Pearson  r = {r_pearson:+.3f}, p-value = {p_pearson:.4f}")
            print(f"  Spearman ρ = {r_spearman:+.3f}, p-value = {p_spearman:.4f}")
            
            # Interpretation
            if p_pearson < 0.001:
                sig_label = "*** (highly significant)"
            elif p_pearson < 0.01:
                sig_label = "** (very significant)"
            elif p_pearson < 0.05:
                sig_label = "* (significant)"
            else:
                sig_label = "(not significant)"
            
            if abs(r_pearson) > 0.7:
                strength = "STRONG"
            elif abs(r_pearson) > 0.5:
                strength = "MODERATE"
            else:
                strength = "WEAK"
            
            print(f"  → {strength} correlation {sig_label}")
    
    # DB-Index correlation (lower is better, so positive correlation expected)
    print(f"\nDavies-Bouldin Index (lower is better):")
    df_db = df_results.groupby('D_dataset')['DB_Index'].mean()
    if len(df_db) > 2:
        d_vals = df_db.index.values
        db_vals = df_db.values
        valid_idx = ~(np.isnan(d_vals) | np.isnan(db_vals))
        d_vals_clean = d_vals[valid_idx]
        db_vals_clean = db_vals[valid_idx]
        
        if len(d_vals_clean) > 2:
            r_pearson_db, p_pearson_db = pearsonr(d_vals_clean, db_vals_clean)
            r_spearman_db, p_spearman_db = spearmanr(d_vals_clean, db_vals_clean)
            
            print(f"  Pearson  r = {r_pearson_db:+.3f}, p-value = {p_pearson_db:.4f}")
            print(f"  Spearman ρ = {r_spearman_db:+.3f}, p-value = {p_spearman_db:.4f}")
            print(f"  → Expected: r > 0 (higher difficulty → higher DB-Index)")
    
    # CH-Index correlation (higher is better, so negative correlation expected)
    print(f"\nCalinski-Harabasz Index (higher is better):")
    df_ch = df_results.groupby('D_dataset')['CH_Index'].mean()
    if len(df_ch) > 2:
        d_vals = df_ch.index.values
        ch_vals = df_ch.values
        valid_idx = ~(np.isnan(d_vals) | np.isnan(ch_vals))
        d_vals_clean = d_vals[valid_idx]
        ch_vals_clean = ch_vals[valid_idx]
        
        if len(d_vals_clean) > 2:
            r_pearson_ch, p_pearson_ch = pearsonr(d_vals_clean, ch_vals_clean)
            r_spearman_ch, p_spearman_ch = spearmanr(d_vals_clean, ch_vals_clean)
            
            print(f"  Pearson  r = {r_pearson_ch:+.3f}, p-value = {p_pearson_ch:.4f}")
            print(f"  Spearman ρ = {r_spearman_ch:+.3f}, p-value = {p_spearman_ch:.4f}")
            print(f"  → Expected: r < 0 (higher difficulty → lower CH-Index)")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("-"*80)
    print("✓ Success indicators:")
    print("  • ARI: r < -0.7 and p < 0.001 (strong negative correlation)")
    print("  • NMI: r < -0.5 and p < 0.001 (moderate-strong negative correlation)")
    print("  • Silhouette: r < -0.3 and p < 0.05 (consistent negative trend)")
    print("  • DB-Index: r > 0.5 and p < 0.001 (higher difficulty → worse clustering)")
    print("  • CH-Index: r < -0.5 and p < 0.001 (higher difficulty → worse clustering)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def table1_global(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the paper's Table 1 *global* rows.

    Important methodological detail: the paper's global correlations are computed
    at the **dataset level** (N = 75 = 15 configs x 5 seeds), i.e. each metric is
    first averaged over the algorithm pool for a given dataset, and the
    correlation with D_dataset is taken over those 75 points. Correlating over
    all (algorithm x dataset) rows instead gives markedly weaker values.
    """
    from scipy.stats import pearsonr, spearmanr

    metrics = ["ARI", "NMI", "Silhouette", "DB_Index", "CH_Index"]
    agg = (df.groupby(["config_id", "seed"])
             .agg({"D_dataset": "first", **{m: "mean" for m in metrics}})
             .reset_index())

    rows = []
    print("\n" + "=" * 78)
    print(f"TABLE 1 — global correlations with D_dataset (dataset level, N={len(agg)})")
    print("=" * 78)
    for m in metrics:
        d = agg[["D_dataset", m]].dropna()
        r, pr = pearsonr(d["D_dataset"], d[m])
        rho, ps = spearmanr(d["D_dataset"], d[m])
        rows.append({"Metric": m, "Pearson_r": r, "Pearson_p": pr,
                     "Spearman_rho": rho, "Spearman_p": ps, "N": len(d)})
        print(f"  {m:11s}: Pearson r = {r:+.3f} (p={pr:.1e})   "
              f"Spearman rho = {rho:+.3f} (p={ps:.1e})   N={len(d)}")

    out = pd.DataFrame(rows)
    path = OUTPUT_DIR / "exp1_table1_global.csv"
    out.to_csv(path, index=False)
    print(f"\n✓ Saved: {path.name}")
    return out


def main():
    """Main entry point.

    Default: run the full 15-config x 5-seed validation sweep (75 datasets).
    With --from-csv: skip the sweep and rebuild the plots and Table 1 from an
    existing exp1_results.csv — e.g. the committed source CSV from R. Riad that
    produced the paper's Section 4.2.
    """
    import argparse
    ap = argparse.ArgumentParser(description="Experiment 1 — D_dataset validation (§4.2).")
    ap.add_argument("--from-csv", type=str, nargs="?", const="__default__", default=None,
                    help="Skip the sweep and rebuild figures/Table 1 from an existing "
                         "exp1_results.csv (defaults to the committed source CSV).")
    ap.add_argument("--criterion", choices=["silhouette", "dbcv", "fixed"], default="silhouette",
                    help="Tuning protocol for the algorithm pool (see clustaudit.audit.algorithms).")
    ap.add_argument("--seeds", type=int, default=None, help="Use only the first N seeds (smoke tests).")
    ap.add_argument("--algos", nargs="*", default=None, help="Run only these algorithms (e.g. --algos DPC); output gets a suffix.")
    ap.add_argument("--no-prerefine", action="store_true", help="Control: keep the raw generated labels (no self-consistency refinement).")
    ap.add_argument("--refine-classifier", choices=["knn", "svm", "random_forest"], default=None,
                    help="Control: classifier used by the pre-noise refinement (default: the one of Theta_amb, k-NN).")
    ap.add_argument("--configs", type=int, default=None, help="Use only the first N configs (smoke tests).")
    ap.add_argument("--regime", choices=["easy", "medium", "hard"], default=None, help="Run one regime only; output gets a suffix.")
    args, _ = ap.parse_known_args()
    _tune_algo.CRITERION = args.criterion
    global ALGOS, NO_PREREFINE, REFINE_CLF
    ALGOS = args.algos
    NO_PREREFINE = args.no_prerefine
    REFINE_CLF = args.refine_classifier
    if args.seeds:
        SEEDS[:] = SEEDS[:args.seeds]
    global REGIME, create_difficulty_configs
    REGIME = args.regime
    if args.regime:
        _all = create_difficulty_configs
        create_difficulty_configs = lambda: [c for c in _all() if c["difficulty"] == args.regime]
    if args.configs:
        _full = create_difficulty_configs
        create_difficulty_configs = lambda: _full()[:args.configs]

    print("\n")

    if args.from_csv is not None:
        csv_path = (OUTPUT_DIR / "exp1_results.csv"
                    if args.from_csv == "__default__" else Path(args.from_csv))
        print("=" * 80)
        print("EXPERIMENT 1 (rebuild from source CSV)")
        print("=" * 80)
        print(f"[from-csv] loading {csv_path}")
        # committed CSV is ';'-separated; a fresh run writes ','. Sniff it.
        df_results = pd.read_csv(csv_path, sep=None, engine="python")
        plot_results(df_results)
        print_summary(df_results)
        table1_global(df_results)
        print("\n" + "=" * 80)
        print("✓ EXPERIMENT 1 (from CSV) COMPLETED")
        print("=" * 80)
        return

    # Run experiment
    df_results = run_experiment()

    # Plot results
    plot_results(df_results)

    # Print summary
    print_summary(df_results)

    # Paper's Table 1 (global, dataset-level)
    table1_global(df_results)

    print("\n" + "=" * 80)
    print("✓ EXPERIMENT 1 COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
