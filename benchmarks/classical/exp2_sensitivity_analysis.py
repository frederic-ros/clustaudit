#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 2: Sensitivity Analysis
===================================

Identifies algorithm-specific sensitivity to individual complexity factors:
- Anisotropy (α): Varies from 1.0 to 10.0
- Noise (η × η_int): 2D grid across total noise and internal fraction
- Dimensionality (d): Varies from 2 to 100

Expected Result: Different algorithms degrade at different critical thresholds.

Author: R. Riad
Date: March 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import copy

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
FIXED_N = False          # set by --fixed-n
SMOKE = False            # set by --smoke
import tune_algo as _tune_algo
OUTPUT_DIR = Path(_HERE) / "results" / "exp2_sensitivity_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# =============================================================================
# BASE CONFIGURATION
# =============================================================================

def get_base_config() -> Dict:
    """Get medium baseline configuration"""
    return {
        'structural': StructuralParams(
            hypercube_connectivity=0.5,
            enable_anisotropy=True,
            anisotropy_factor_range=(1.5, 2.5),
            enable_rotation=True,
            cluster_std_range=(0.15, 0.25),
            cluster_size_imbalance=1.3,
            density_params=DensityParams(
                inter_density_variation=0.3,
                intra_density_strength=0.5,
            ),
        ),
        'noise': NoiseParams(noise_ratio=0.1),
        'ambiguity': AmbiguityParams(
            enable_pre_refinement=True,
            enable_post_refinement=True,
        ),
    }


# =============================================================================
# EXPERIMENT: ANISOTROPY SENSITIVITY
# =============================================================================

def run_anisotropy_sensitivity() -> pd.DataFrame:
    """Test algorithm sensitivity to anisotropy"""
    print("\n" + "-"*80)
    print("ANISOTROPY SENSITIVITY (α: 1.0 → 10.0)")
    print("-"*80)
    
    base_config = get_base_config()
    results = []
    
    #ALPHA_VALUES = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    # Vary anisotropy α from 1 to 10
    ALPHA_VALUES = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] if not SMOKE else [1.0, 5.0]
    
    for alpha in ALPHA_VALUES:
        for seed in SEEDS:
            print(f"  α={alpha}, seed={seed}")
            
            try:
                config = copy.deepcopy(base_config)
                config['structural'].anisotropy_factor_range = (alpha, alpha + 0.5)
                
                state = generate_dataset(
                    n_samples=1000,
                    dimension=20,
                    n_clusters=5,
                    structural=config['structural'],
                    noise=config['noise'],
                    ambiguity=config['ambiguity'],
                    random_seed=seed,
                )
                
                X, y_true = state.Z, state.y
                prof = profile_from_state(state, seed=seed)
                complexity = prof.score
                prof_cols = {k: v for k, v in prof.to_dict().items()
                             if k not in ('D_dataset', 'n_samples', 'dimension', 'n_clusters')}
                n_clusters_true = len(np.unique(y_true))
                
                algorithms = {
                    "KMeans": tune_kmeans(X, seed),
                    "DBSCAN": tune_dbscan(X),
                    "HDBSCAN": tune_hdbscan(X),
                    "OPTICS": tune_optics(X),
                    "Ward": tune_agglomerative(X, "ward"),
                    "Average": tune_agglomerative(X, "average"),
                    "Spectral": tune_spectral(X, seed),
                    "GMM": tune_gmm(X, seed),
                }
                
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
                        'parameter': 'anisotropy',
                        'value': alpha,
                        'seed': seed,
                        'algorithm': algo_name,
                        'D_dataset': complexity,
                        **prof_cols,
                        'n_clusters_true': n_clusters_true,
                        'n_clusters_found': res['n_clusters_found'],
                        **res,
                    })
            except Exception as e:
                print(f"    ✗ Failed: {str(e)}")
    
    return pd.DataFrame(results)


# =============================================================================
# EXPERIMENT: NOISE SENSITIVITY
# =============================================================================

def run_noise_sensitivity() -> pd.DataFrame:
    """Test algorithm sensitivity to noise (2D grid: total noise × internal fraction)"""
    print("\n" + "-"*80)
    print("NOISE SENSITIVITY (η: 0.0 → 0.50, η_int: 0.0 → 1.0)")
    print("-"*80)
    
    base_config = get_base_config()
    results = []
    
    # Vary total noise η from 0 to 0.5
    ETA_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50] if not SMOKE else [0.1]
    
    # Also vary internal fraction η_int
    ETA_INT_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0] if not SMOKE else [0.0, 1.0]
    
    # Generate grid: η × η_int
    for eta in ETA_VALUES:
        for eta_int in ETA_INT_VALUES:
            for seed in SEEDS:  # Sample seeds for efficiency (2D grid is large)
                print(f"  η={eta}, η_int={eta_int}, seed={seed}")
                
                try:
                    config = copy.deepcopy(base_config)
                    config['noise'].noise_ratio = eta
                    # v1 BUG: the attribute was tested as `internal_fraction` (which does
                    # not exist), so eta_int was never applied and all five levels ran on
                    # identical data. The field is `internal_noise_ratio`.
                    config['noise'].internal_noise_ratio = eta_int
                    
                    state = generate_dataset(
                        n_samples=1000,
                        dimension=20,
                        n_clusters=5,
                        structural=config['structural'],
                        noise=config['noise'],
                        ambiguity=config['ambiguity'],
                        random_seed=seed,
                    )
                    
                    X, y_true = state.Z, state.y
                    prof = profile_from_state(state, seed=seed)
                    complexity = prof.score
                    prof_cols = {k: v for k, v in prof.to_dict().items()
                                 if k not in ('D_dataset', 'n_samples', 'dimension', 'n_clusters')}
                    n_clusters_true = len(np.unique(y_true))
                    
                    algorithms = {
                        "KMeans": tune_kmeans(X, seed),
                        "DBSCAN": tune_dbscan(X),
                        "HDBSCAN": tune_hdbscan(X),
                        "OPTICS": tune_optics(X),
                        "Ward": tune_agglomerative(X, "ward"),
                        "Average": tune_agglomerative(X, "average"),
                        "Spectral": tune_spectral(X, seed),
                        "GMM": tune_gmm(X, seed),
                    }
                    
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
                            'parameter': 'noise',
                            'value_eta': eta,
                            'value_eta_int': eta_int,
                            'seed': seed,
                            'algorithm': algo_name,
                            'D_dataset': complexity,
                            **prof_cols,
                            'n_clusters_true': n_clusters_true,
                            'n_clusters_found': res['n_clusters_found'],
                            **res,
                        })
                except Exception as e:
                    print(f"    ✗ Failed: {str(e)}")
    
    return pd.DataFrame(results)


# =============================================================================
# EXPERIMENT: DIMENSIONALITY SENSITIVITY
# =============================================================================

def run_dimensionality_sensitivity() -> pd.DataFrame:
    """Test algorithm sensitivity to dimensionality (curse of dimensionality)"""
    print("\n" + "-"*80)
    print("DIMENSIONALITY SENSITIVITY (d: 2 → 200)")
    print("-"*80)
    
    base_config = get_base_config()
    results = []
    
    # Vary dimension d from 2 to 100
    D_VALUES = [3, 5, 10, 15, 20, 30, 50, 70, 100] if not SMOKE else [5, 20]
    
    for d in D_VALUES:
        for seed in SEEDS:  # Sample seeds for efficiency
            print(f"  d={d}, seed={seed}")
            
            try:
                # n ∝ d, or fixed n as a control (--fixed-n)
                n = 1000 if FIXED_N else max(1000, 50 * d)
                config = copy.deepcopy(base_config)
                
                state = generate_dataset(
                    n_samples=n,
                    dimension=d,
                    n_clusters=5,
                    structural=config['structural'],
                    noise=config['noise'],
                    ambiguity=config['ambiguity'],
                    random_seed=seed,
                )
                
                X, y_true = state.Z, state.y
                prof = profile_from_state(state, seed=seed)
                complexity = prof.score
                prof_cols = {k: v for k, v in prof.to_dict().items()
                             if k not in ('D_dataset', 'n_samples', 'dimension', 'n_clusters')}
                n_clusters_true = len(np.unique(y_true))
                
                algorithms = {
                    "KMeans": tune_kmeans(X, seed),
                    "DBSCAN": tune_dbscan(X),
                    "HDBSCAN": tune_hdbscan(X),
                    "OPTICS": tune_optics(X),
                    "Ward": tune_agglomerative(X, "ward"),
                    "Average": tune_agglomerative(X, "average"),
                    "Spectral": tune_spectral(X, seed),
                    "GMM": tune_gmm(X, seed),
                }
                
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
                        'parameter': 'dimensionality',
                        'value': d,
                        'seed': seed,
                        'algorithm': algo_name,
                        'D_dataset': complexity,
                        **prof_cols,
                        'n_clusters_true': n_clusters_true,
                        'n_clusters_found': res['n_clusters_found'],
                        'n_samples': n,
                        **res,
                    })
            except Exception as e:
                print(f"    ✗ Failed: {str(e)}")
    
    return pd.DataFrame(results)


# =============================================================================
# PLOTTING & ANALYSIS
# =============================================================================

def plot_results(df_results: pd.DataFrame):
    """Create sensitivity plots (1D for anisotropy/dimensionality, 2D heatmaps for noise)"""
    print("\nGenerating sensitivity plots...")
    
    # Handle noise separately (2D heatmap)
    noise_subset = df_results[df_results['parameter'] == 'noise']
    if len(noise_subset) > 0:
        # Generate individual heatmap for each algorithm
        for algo in sorted(noise_subset['algorithm'].unique()):
            algo_data = noise_subset[noise_subset['algorithm'] == algo]
            
            # Pivot table: η (rows) × η_int (columns), ARI as values
            ari_pivot = algo_data.pivot_table(values='ARI', index='value_eta', columns='value_eta_int', aggfunc='mean')
            
            # Create large, focused ARI heatmap
            fig, ax = plt.subplots(figsize=(10, 7))
            
            sns.heatmap(ari_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=ax, vmin=-0.1, vmax=1.0, cbar_kws={'label': 'ARI Score'},
                       linewidths=0.5, linecolor='gray')
            ax.set_xlabel('Internal Fraction (η_int)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total Noise (η)', fontsize=12, fontweight='bold')
            ax.set_title(f'{algo}: ARI Performance across Noise Grid (η × η_int)', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"sensitivity_noise_{algo.lower()}_ari_heatmap.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: sensitivity_noise_{algo.lower()}_ari_heatmap.pdf")
        
        # Generate combined heatmap showing all algorithms in subplots
        unique_algos = sorted(noise_subset['algorithm'].unique())
        n_algos = len(unique_algos)
        n_cols = min(3, n_algos)
        n_rows = (n_algos + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        if n_algos == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, algo in enumerate(unique_algos):
            algo_data = noise_subset[noise_subset['algorithm'] == algo]
            ari_pivot = algo_data.pivot_table(values='ARI', index='value_eta', columns='value_eta_int', aggfunc='mean')
            
            sns.heatmap(ari_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=axes[idx], vmin=-0.1, vmax=1.0, cbar_kws={'label': 'ARI'},
                       linewidths=0.5, linecolor='gray')
            axes[idx].set_xlabel('η_int', fontsize=11)
            axes[idx].set_ylabel('η', fontsize=11)
            axes[idx].set_title(f'{algo}', fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(unique_algos), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('ARI Performance across Noise Grid (η × η_int) - All Algorithms', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"sensitivity_noise_all_algorithms_ari_grid.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: sensitivity_noise_all_algorithms_ari_grid.pdf")
    
    # Handle 1D parameters (anisotropy, dimensionality)
    for param in ['anisotropy', 'dimensionality']:
        subset = df_results[df_results['parameter'] == param]
        if len(subset) == 0:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, metric in enumerate(['ARI', 'NMI']):
            ax = axes[idx]
            
            for algo in sorted(subset['algorithm'].unique()):
                data = subset[subset['algorithm'] == algo].groupby('value')[metric].agg(['mean', 'std'])
                
                ax.plot(data.index, data['mean'], marker='o', label=algo, linewidth=2, markersize=8)
                ax.fill_between(data.index, 
                               data['mean'] - data['std'], 
                               data['mean'] + data['std'], 
                               alpha=0.2)
            
            ax.set_xlabel(f'{param.capitalize()}', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric} vs {param.capitalize()}', fontsize=12)
            ax.legend(fontsize=9, loc='best')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"sensitivity_{param}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: sensitivity_{param}.pdf")


def print_summary(df_results: pd.DataFrame):
    """Print critical thresholds and comprehensive sensitivity analysis"""
    print("\n" + "="*80)
    print("CRITICAL THRESHOLDS & SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Handle noise separately (2D grid)
    noise_subset = df_results[df_results['parameter'] == 'noise']
    if len(noise_subset) > 0:
        print(f"\nNOISE (2D Grid: η × η_int):")
        print("-" * 80)
        
        for algo in sorted(noise_subset['algorithm'].unique()):
            algo_data = noise_subset[noise_subset['algorithm'] == algo]
            
            # Pivot to get (eta, eta_int) grid
            ari_pivot = algo_data.pivot_table(
                values='ARI', 
                index='value_eta', 
                columns='value_eta_int', 
                aggfunc='mean'
            )
            
            print(f"\n  {algo}:")
            print(f"    ARI over η × η_int grid:")
            print(f"    (η in columns: {list(ari_pivot.columns)}, η in rows: {list(ari_pivot.index)})")
            
            for eta in sorted(ari_pivot.index):
                ari_vals = ari_pivot.loc[eta].values
                ari_str = " | ".join([f"{v:.2f}" for v in ari_vals])
                print(f"      η={eta:4.2f}: {ari_str}")
            
            # Find critical points
            avg_ari = algo_data['ARI'].mean()
            min_ari = algo_data['ARI'].min()
            print(f"    Mean ARI: {avg_ari:.3f}, Min ARI: {min_ari:.3f}")
            
            # Count low performance (ARI < 0.5)
            low_perf = len(algo_data[algo_data['ARI'] < 0.5])
            total = len(algo_data)
            print(f"    Low performance cases (ARI < 0.5): {low_perf}/{total}")
    
    # Handle other parameters (1D)
    for param in ['anisotropy', 'dimensionality']:
        subset = df_results[df_results['parameter'] == param]
        if len(subset) == 0:
            continue
        
        print(f"\n{param.upper()}:")
        print("-" * 80)
        
        for algo in sorted(subset['algorithm'].unique()):
            algo_data = subset[subset['algorithm'] == algo]
            
            # Aggregate metrics by parameter value
            metrics_by_value = algo_data.groupby('value')[['ARI', 'NMI', 'Silhouette', 'DB_Index', 'CH_Index']].mean()
            
            ari_range = f"[{metrics_by_value['ARI'].min():.3f}, {metrics_by_value['ARI'].max():.3f}]"
            nmi_range = f"[{metrics_by_value['NMI'].min():.3f}, {metrics_by_value['NMI'].max():.3f}]"
            
            print(f"\n  {algo}:")
            print(f"    ARI range:       {ari_range}")
            print(f"    NMI range:       {nmi_range}")
            print(f"    Avg D_dataset:   {algo_data['D_dataset'].mean():.3f}")
            print(f"    Avg clusters (true/found): {algo_data['n_clusters_true'].mean():.1f}/{algo_data['n_clusters_found'].mean():.1f}")
            print(f"    DB-Index mean:   {metrics_by_value['DB_Index'].mean():.3f} (lower better)")
            print(f"    CH-Index mean:   {metrics_by_value['CH_Index'].mean():.1f} (higher better)")
            
            # Find degradation point (where ARI < 0.5)
            ari_vals = metrics_by_value['ARI']
            degradation_point = ari_vals[ari_vals < 0.5]
            if len(degradation_point) > 0:
                critical_value = degradation_point.index[0]
                print(f"    Critical {param} threshold: {critical_value:.2f} (ARI < 0.5)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point.

    By default, runs the full sensitivity sweeps (anisotropy, noise,
    dimensionality) and then plots/summarizes. With --from-csv, it skips the
    (long) sweeps and regenerates the figures/summary from an existing
    exp2_results.csv — e.g. the one committed under results/ that backs the
    paper's Section 4.3.
    """
    import argparse
    ap = argparse.ArgumentParser(description="Experiment 2 — sensitivity analysis (§4.3).")
    ap.add_argument("--from-csv", type=str, nargs="?", const="__default__",
                    default=None,
                    help="Skip the sweeps and replot from an existing exp2_results.csv. "
                         "With no path, uses results/exp2_sensitivity_analysis/exp2_results.csv.")
    ap.add_argument("--fixed-n", action="store_true",
                    help="Dimensionality sweep with n fixed at 1000 instead of n = max(1000, 50 d).")
    ap.add_argument("--criterion", choices=["silhouette", "dbcv", "fixed"], default="silhouette",
                    help="Tuning protocol for the algorithm pool (see clustaudit.audit.algorithms).")
    ap.add_argument("--out", default=None,
                    help="name of the result file in results/exp2_sensitivity_analysis/ (default: derived "
                         "from --criterion, --fixed-n and --only); scripts/reproduce.py always passes it")
    ap.add_argument("--only", nargs="*", choices=["anisotropy", "noise", "dimensionality"],
                    help="Run a subset of the sweeps.")
    ap.add_argument("--smoke", action="store_true", help="Tiny grids, one seed (pipeline check).")
    args, _ = ap.parse_known_args()
    global FIXED_N
    FIXED_N = args.fixed_n
    _tune_algo.CRITERION = args.criterion
    global SEEDS, SMOKE
    SMOKE = args.smoke
    if args.smoke:
        SEEDS = SEEDS[:1]
    tag = f"_{args.criterion}" + ("_fixedn" if args.fixed_n else "")
    if args.only:   # one file per sub-sweep so that successive runs never overwrite each other
        tag += "_" + "_".join(sorted(args.only))

    print("="*80)
    print("EXPERIMENT 2: SENSITIVITY ANALYSIS")
    print("="*80)

    if args.from_csv is not None:
        csv_path = (OUTPUT_DIR / "exp2_results.csv"
                    if args.from_csv == "__default__" else Path(args.from_csv))
        print(f"[from-csv] loading {csv_path}")
        df_results = pd.read_csv(csv_path)
        plot_results(df_results)
        print_summary(df_results)
        print("\n" + "="*80)
        print("✓ EXPERIMENT 2 (replot from CSV) COMPLETED")
        print("="*80)
        return

    results_all = []
    only = set(args.only or ["anisotropy", "noise", "dimensionality"])
    if "anisotropy" in only:
        results_all.append(run_anisotropy_sensitivity())
    if "noise" in only:
        results_all.append(run_noise_sensitivity())
    if "dimensionality" in only:
        results_all.append(run_dimensionality_sensitivity())

    df_results = pd.concat(results_all, ignore_index=True)
    df_results["criterion"] = args.criterion
    df_results["fixed_n"] = args.fixed_n
    out_csv = OUTPUT_DIR / (args.out or ("exp2_results.csv" if tag == "_silhouette" else f"exp2_results{tag}.csv"))
    df_results.to_csv(out_csv, index=False)
    print(f"\n✓ Results saved: {out_csv}")

    # Plot and summarize
    plot_results(df_results)
    print_summary(df_results)

    print("\n" + "="*80)
    print("✓ EXPERIMENT 2 COMPLETED")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
