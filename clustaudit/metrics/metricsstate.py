# -*- coding: utf-8 -*-
"""
Clustering Metrics Utilities
----------------------------

Compute enriched clustering metrics including density-aware indicators,
save them, and print a compact summary.

Author: frederic.ros
"""

import os
import csv
from datetime import datetime
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist

from ..core.state import ClusteringState
from .complexity import compute_complexity_score


# ===================================================================
# 1. UTILS
# ===================================================================
def _normalize(x, min_val=0.0, max_val=1.0):
    """Safe normalization to [0,1]."""
    if np.isnan(x):
        return 0.0
    return np.clip((x - min_val) / (max_val - min_val + 1e-12), 0.0, 1.0)


# ===================================================================
# 2. COMPUTE COMPLETE METRICS
# ===================================================================
# -*- coding: utf-8 -*-
"""
Clustering Metrics Utilities - Refactored
-----------------------------------------
Compute enriched clustering metrics with density-aware indicators.
Author: frederic.ros
"""

import os
import csv
from datetime import datetime
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist

from ..core.state import ClusteringState
from .complexity import compute_complexity_score


# ----------------------------
# --- Helper functions ---
# ----------------------------

def _normalize(x, min_val=0.0, max_val=1.0):
    if np.isnan(x):
        return 0.0
    return np.clip((x - min_val) / (max_val - min_val + 1e-12), 0.0, 1.0)

def _compute_intra_cluster(Zk_core, eps=1e-6):
    """Compute elongation, compactness, intra_density."""
    if len(Zk_core) > 1:
        dists = pdist(Zk_core)
        compactness = np.mean(dists)
        intra_density_mean = 1.0 / (compactness + eps)
        intra_density_std = np.std(dists)

        pca = PCA(n_components=min(2, Zk_core.shape[1]))
        pca.fit(Zk_core)
        var_ratio = pca.explained_variance_ratio_
        elongation = var_ratio[0] / max(var_ratio[1] if len(var_ratio) > 1 else eps, eps)
    else:
        elongation = np.nan
        compactness = np.nan
        intra_density_mean = np.nan
        intra_density_std = np.nan
    return elongation, compactness, intra_density_mean, intra_density_std

def _compute_density_contrast(Zk_core, Zk_boundary, eps=1e-6):
    """Compute Δρ for a cluster."""
    if len(Zk_core) > 0:
        rho_core = 1.0 / (np.mean(cdist(Zk_core, Zk_core), axis=1) + eps)
    else:
        rho_core = np.array([1.0])

    if len(Zk_boundary) > 0:
        rho_boundary = 1.0 / (np.mean(cdist(Zk_boundary, Zk_boundary), axis=1) + eps)
        delta_rho = np.mean(rho_core) / (np.mean(rho_boundary) + eps)
    else:
        delta_rho = np.max(rho_core)  # facile si pas de boundary
    return delta_rho, rho_boundary.tolist() if len(Zk_boundary) > 0 else []

def _compute_boundary_metrics(Zk_core, Zk_boundary, Z_other_core, eps=1e-6):
    """Compute boundary metrics per cluster."""
    if len(Zk_core) > 0 and len(Zk_boundary) > 0:
        dist_boundary_core = cdist(Zk_boundary, Zk_core)
        min_dists = np.min(dist_boundary_core, axis=1)
        boundary_core_distance_mean = np.mean(min_dists)
        boundary_core_distance_std = np.std(min_dists)
        boundary_density = len(Zk_boundary) / (boundary_core_distance_mean + eps)
    else:
        boundary_core_distance_mean = np.nan
        boundary_core_distance_std = np.nan
        boundary_density = np.nan

    if len(Zk_boundary) > 0 and len(Z_other_core) > 0:
        cross_dist = cdist(Zk_boundary, Z_other_core)
        cross_cluster_proximity_mean = np.mean(np.min(cross_dist, axis=1))
        cross_cluster_proximity_std = np.std(np.min(cross_dist, axis=1))
    else:
        cross_cluster_proximity_mean = np.nan
        cross_cluster_proximity_std = np.nan

    return {
        'boundary_density': boundary_density,
        'boundary_core_distance_mean': boundary_core_distance_mean,
        'boundary_core_distance_std': boundary_core_distance_std,
        'cross_cluster_proximity_mean': cross_cluster_proximity_mean,
        'cross_cluster_proximity_std': cross_cluster_proximity_std
    }

def _compute_O_rho(state, unique_clusters, core_idx, boundary_idx, eps=1e-6):
    """Compute dataset-level inter-cluster density Oρ."""
    if len(boundary_idx) == 0:
        return 1.0

    Z_core = state.Z[core_idx]
    y_core = state.y[core_idx]
    all_boundary_idx = boundary_idx
    y_boundary = state.y[all_boundary_idx]

    O_rho_list = []
    for i, idx in enumerate(all_boundary_idx):
        cluster_i = y_boundary[i]
        if not np.any(y_core == cluster_i):
            continue  # a cluster with no core point (all boundary): nothing to contrast against
        rho_i = 1.0 / (np.mean(cdist([state.Z[idx]], Z_core[y_core == cluster_i]), axis=1)[0] + eps)
        other_clusters = [c for c in unique_clusters if c != cluster_i and np.any(y_core == c)]
        if other_clusters:
            nn_cluster = min(other_clusters, key=lambda c: np.linalg.norm(
                np.mean(Z_core[y_core == cluster_i], axis=0) - np.mean(Z_core[y_core == c], axis=0)
            ))
            rho_nn = 1.0 / (np.mean(cdist(Z_core[y_core == nn_cluster], Z_core[y_core == nn_cluster]), axis=1).mean() + eps)
            rho_min = min(np.mean(rho_i), rho_nn)
            O_rho_list.append(rho_i / rho_min)
    return np.mean(O_rho_list) if O_rho_list else 1.0


# ----------------------------
# --- Main function ---
# ----------------------------

def compute_complete_metrics(state: "ClusteringState", include_noise=True, eps=1e-6):
    metrics = {'global': {}, 'clusters': {}, 'boundary': {}, 'noise': {}}

    core_idx = state.core_points()
    boundary_idx = state.boundary_points()
    noise_idx = state.noise_points()

    Z_core = state.Z[core_idx] if len(core_idx) else np.empty((0, state.dimension()))
    y_core = state.y[core_idx] if len(core_idx) else np.empty(0, dtype=int)

    # --- Global metrics ---
    metrics['global'].update({
        'n_core': len(core_idx),
        'n_boundary': len(boundary_idx),
        'n_noise': len(noise_idx),
        'mean_entropy': np.mean(state.entropy[core_idx]) if len(core_idx) else np.nan,
        'mean_confidence': np.mean(state.confidence[core_idx]) if len(core_idx) else np.nan
    })

    n_cls = len(np.unique(y_core))
    if 1 < n_cls < len(core_idx):   # silhouette needs 2 <= n_labels <= n_samples - 1
        metrics['global'].update({
            'silhouette': silhouette_score(Z_core, y_core),
            'calinski_harabasz': calinski_harabasz_score(Z_core, y_core),
            'davies_bouldin': davies_bouldin_score(Z_core, y_core)
        })
    else:
        metrics['global'].update({'silhouette': np.nan, 'calinski_harabasz': np.nan, 'davies_bouldin': np.nan})

    unique_clusters = np.unique(y_core)
    all_rho_boundary = []

    for k in unique_clusters:
        idx_k_core = core_idx[state.y[core_idx] == k]
        idx_k_boundary = boundary_idx[state.y[boundary_idx] == k] if len(boundary_idx) else np.array([], dtype=int)
        Zk_core = state.Z[idx_k_core] if len(idx_k_core) else np.empty((0, state.dimension()))
        Zk_boundary = state.Z[idx_k_boundary] if len(idx_k_boundary) else np.empty((0, state.dimension()))

        # --- Cluster intra metrics ---
        elongation, compactness, intra_density_mean, intra_density_std = _compute_intra_cluster(Zk_core, eps)

        # --- Density contrast ---
        delta_rho, rho_boundary = _compute_density_contrast(Zk_core, Zk_boundary, eps)
        all_rho_boundary.extend(rho_boundary)

        # --- Boundary metrics ---
        other_core_idx = core_idx[state.y[core_idx] != k]
        metrics['boundary'][k] = _compute_boundary_metrics(Zk_core, Zk_boundary, state.Z[other_core_idx], eps)

        size_core = len(idx_k_core)
        size_boundary = len(idx_k_boundary)
        boundary_ratio = size_boundary / max(size_core + size_boundary, 1)

        metrics['clusters'][k] = {
            'size_core': size_core,
            'size_boundary': size_boundary,
            'boundary_ratio': boundary_ratio,
            'mean_entropy': np.mean(state.entropy[idx_k_core]) if size_core else np.nan,
            'mean_confidence': np.mean(state.confidence[idx_k_core]) if size_core else np.nan,
            'elongation': elongation,
            'compactness': compactness,
            'intra_density_mean': intra_density_mean,
            'intra_density_std': intra_density_std,
            'delta_rho': delta_rho
        }

    # --- Dataset-level inter-cluster density ---
    metrics['global']['O_rho'] = _compute_O_rho(state, unique_clusters, core_idx, boundary_idx, eps)

    # --- Noise metrics ---
    if include_noise and len(noise_idx) > 0 and len(core_idx) > 0:
        Z_noise = state.Z[noise_idx]
        distances = cdist(Z_noise, Z_core)
        dist_to_core = np.min(distances, axis=1)
        metrics['noise'] = {
            'noise_ratio': len(noise_idx) / max(len(state.y), 1),
            'internal_noise_ratio': np.sum(state.masks['internal_noise']) / max(len(state.y), 1),
            'distant_noise_ratio': np.sum(state.masks['distant_noise']) / max(len(state.y), 1),
            'distance_noise_to_nearest_core_mean': np.mean(dist_to_core),
            'distance_noise_to_nearest_core_std': np.std(dist_to_core)
        }
    else:
        metrics['noise'] = {
            'noise_ratio': 0.0,
            'internal_noise_ratio': 0.0,
            'distant_noise_ratio': 0.0,
            'distance_noise_to_nearest_core_mean': np.nan,
            'distance_noise_to_nearest_core_std': np.nan
        }

    metrics = finalize_global_metrics(metrics)
    complexity = compute_complexity_score(metrics)
    metrics['global']['complexity_score'] = complexity['complexity_score']
    metrics['global']['complexity_components'] = complexity['components']

    return metrics


# ===================================================================
# 3. Metrics with/without noise
# ===================================================================
def compute_metrics_with_and_without_noise(state: "ClusteringState"):
    metrics_with_noise = compute_complete_metrics(state, include_noise=True)

    all_idx = np.where(~state.masks['noise'])[0]
    filtered_state = ClusteringState(
        Z=state.Z[all_idx],
        y=state.y[all_idx],
        masks={k: v[all_idx] for k, v in state.masks.items()},
        entropy=state.entropy[all_idx],
        confidence=state.confidence[all_idx],
        metadata=state.metadata.copy(),
        timestamp=state.timestamp,
        random_seed=state.random_seed
    )

    metrics_without_noise = compute_complete_metrics(filtered_state, include_noise=False)

    return {
        'with_noise': metrics_with_noise,
        'without_noise': metrics_without_noise
    }


# ===================================================================
# 4. Save metrics
# ===================================================================
def save_metrics(metrics_dict, output_dir="metrics_output", filename_base=None, save_csv=True, save_txt=True):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = filename_base or f"metrics_{timestamp}"

    for variant, metrics in metrics_dict.items():

        if save_csv:
            csv_path = os.path.join(output_dir, f"{base}_{variant}.csv")
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Global
                writer.writerow(["SECTION", "GLOBAL"])
                for k, v in metrics.get("global", {}).items():
                    writer.writerow([k, v])
                writer.writerow([])
                # Clusters
                writer.writerow(["SECTION", "CLUSTERS"])
                clusters = metrics.get("clusters", {})
                if clusters:
                    header = ["cluster_id"] + list(next(iter(clusters.values())).keys())
                    writer.writerow(header)
                    for cid, c_metrics in clusters.items():
                        writer.writerow([cid] + [c_metrics.get(h, None) for h in header[1:]])
                writer.writerow([])
                # Boundary
                writer.writerow(["SECTION", "BOUNDARY"])
                boundary = metrics.get("boundary", {})
                if boundary:
                    header = ["cluster_id"] + list(next(iter(boundary.values())).keys())
                    writer.writerow(header)
                    for cid, b_metrics in boundary.items():
                        writer.writerow([cid] + [b_metrics.get(h, None) for h in header[1:]])
                writer.writerow([])
                # Noise
                writer.writerow(["SECTION", "NOISE"])
                for k, v in metrics.get("noise", {}).items():
                    writer.writerow([k, v])

        if save_txt:
            txt_path = os.path.join(output_dir, f"{base}_{variant}.txt")
            with open(txt_path, mode="w", encoding="utf-8") as f:
                f.write(f"METRICS REPORT — {variant.upper()}\n")
                f.write("=" * 70 + "\n\n")
                f.write("GLOBAL METRICS\n" + "-"*30 + "\n")
                for k, v in metrics.get("global", {}).items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
                f.write("CLUSTER METRICS\n" + "-"*30 + "\n")
                for cid, c_metrics in metrics.get("clusters", {}).items():
                    f.write(f"Cluster {cid}\n")
                    for k, v in c_metrics.items():
                        f.write(f"  {k}: {v}\n")
                    f.write("\n")
                f.write("BOUNDARY METRICS\n" + "-"*30 + "\n")
                for cid, b_metrics in metrics.get("boundary", {}).items():
                    f.write(f"Cluster {cid}\n")
                    for k, v in b_metrics.items():
                        f.write(f"  {k}: {v}\n")
                    f.write("\n")
                f.write("NOISE METRICS\n" + "-"*30 + "\n")
                for k, v in metrics.get("noise", {}).items():
                    f.write(f"{k}: {v}\n")

    print(f"✓ Metrics saved in '{output_dir}' (CSV={save_csv}, TXT={save_txt})")


# ===================================================================
# 5. Global stats helper
# ===================================================================
def finalize_global_metrics(metrics):
    clusters = metrics["clusters"]
    global_m = metrics["global"]

    total_core = sum(c["size_core"] for c in clusters.values())
    total_boundary = sum(c.get("size_boundary", 0) for c in clusters.values())
    total_noise = global_m.get("n_noise", 0)

    total_samples = total_core + total_boundary + total_noise
    global_m.setdefault("n_samples", total_samples)
    global_m.setdefault("boundary_ratio", total_boundary / max(total_samples, 1))
    global_m.setdefault("noise_ratio", total_noise / max(total_samples, 1))

    return metrics


# ===================================================================
# 6. Cluster difficulty
# ===================================================================
def compute_cluster_difficulty(cluster_metrics, global_stats, weights=None):
    weights = weights or {
        "boundary_ratio": 0.25,
        "elongation": 0.20,
        "compactness": 0.20,
        "entropy": 0.10,
        "size": 0.10,
        "delta_rho": 0.15
    }
    eps = 1e-6

    boundary = cluster_metrics.get("boundary_ratio", 0.0)
    elong = cluster_metrics.get("elongation", 1.0) / max(global_stats.get("elongation_mean", 1.0), eps)
    compact = cluster_metrics.get("compactness", 1.0) / max(global_stats.get("compactness_mean", 1.0), eps)
    entropy = cluster_metrics.get("mean_entropy", 0.0) / max(global_stats.get("entropy_mean", 1.0), eps)
    size_penalty = np.clip(global_stats.get("size_mean", 1.0) / (cluster_metrics.get("size_core", 1) + eps), 0, 2.0)
    delta_rho = cluster_metrics.get("delta_rho", 1.0)

    components = {
        "boundary_ratio": boundary,
        "elongation": elong,
        "compactness": compact,
        "entropy": entropy,
        "size_penalty": size_penalty,
        "delta_rho": delta_rho
    }

    score = (
        weights["boundary_ratio"] * boundary +
        weights["elongation"] * elong +
        weights["compactness"] * compact +
        weights["entropy"] * entropy +
        weights["size"] * size_penalty +
        weights["delta_rho"] * delta_rho
    )

    return score, components


# ===================================================================
# 7. Global cluster stats
# ===================================================================
def compute_global_cluster_stats(clusters):
    sizes = np.array([c.get("size_core", 0) for c in clusters.values()])
    elong = np.array([c.get("elongation", np.nan) for c in clusters.values()])
    compact = np.array([c.get("compactness", np.nan) for c in clusters.values()])
    entropy = np.array([c.get("mean_entropy", np.nan) for c in clusters.values()])
    delta_rho = np.array([c.get("delta_rho", np.nan) for c in clusters.values()])

    return {
        "size_mean": np.nanmean(sizes),
        "elongation_mean": np.nanmean(elong) if np.any(~np.isnan(elong)) else 1.0,
        "compactness_mean": np.nanmean(compact) if np.any(~np.isnan(compact)) else 1.0,
        "entropy_mean": np.nanmean(entropy) if np.any(~np.isnan(entropy)) else 0.0,
        "delta_rho_mean": np.nanmean(delta_rho) if np.any(~np.isnan(delta_rho)) else 1.0
    }


# ===================================================================
# 8. Print cluster summary
# ===================================================================
def print_cluster_summary(metrics, key="with_noise", weights=None):
    data = metrics[key]
    clusters = data["clusters"]
    global_m = data["global"]
    global_stats = compute_global_cluster_stats(clusters)

    print(f"\n=== CLUSTER SYNTHESIS ({key.upper()}) ===")
    print(
        f"Samples={global_m.get('n_samples','?')} | "
        f"Clusters={len(clusters)} | "
        f"Noise ratio={global_m.get('noise_ratio',0):.2f} | "
        f"Boundary ratio={global_m.get('boundary_ratio',0):.2f} | "
        f"Oρ={global_m.get('O_rho',np.nan):.3f}"
    )

    if "mean_entropy" in global_m:
        print(
            f"Entropy μ={global_m['mean_entropy']:.3f} | "
            f"Confidence μ={global_m.get('mean_confidence',0):.3f}"
        )

    # Compute difficulty
    scores = {}
    for cid, c in clusters.items():
        score, comps = compute_cluster_difficulty(c, global_stats, weights)
        scores[cid] = score
        clusters[cid]["difficulty_score"] = score
        clusters[cid]["difficulty_components"] = comps

    scores_arr = np.array(list(scores.values()))
    mean_score = scores_arr.mean() if len(scores_arr) > 0 else 0.0
    std_score = scores_arr.std() if len(scores_arr) > 0 else 0.0

    # Categorize clusters
    upper = mean_score + 1.25 * std_score
    lower = mean_score - 1.25 * std_score
    hard   = [cid for cid, s in scores.items() if s > upper]
    easy   = [cid for cid, s in scores.items() if s < lower]
    medium = [cid for cid, s in scores.items() if lower <= s <= upper]

    # Print summary
    print(
        f"Difficulty μ={mean_score:.2f} | σ={std_score:.2f} | "
        f"min={scores_arr.min() if len(scores_arr)>0 else np.nan:.2f} | "
        f"max={scores_arr.max() if len(scores_arr)>0 else np.nan:.2f}"
    )
    if hard:
        print(f"🔥 Hard clusters: {hard}")
    if medium:
        print(f"🟡 Medium clusters: {medium}")
    if easy:
        print(f"🟢 Easy clusters: {easy}")

    print("\n--- CLUSTER DETAILS ---")
    for cid, c in clusters.items():
        elong_str = f"{c['elongation']:.2f}" if not np.isnan(c['elongation']) else "?"
        comp_str = f"{c['compactness']:.3f}" if not np.isnan(c['compactness']) else "?"
        delta_rho_str = f"{c.get('delta_rho', np.nan):.2f}"
        diff_str = f"{c.get('difficulty_score', np.nan):.2f}"
        print(
            f"C{cid}: size={c['size_core']:4d} | bound={c['boundary_ratio']:.2f} | "
            f"elong={elong_str} | comp={comp_str} | Δρ={delta_rho_str} | diff={diff_str}"
        )
