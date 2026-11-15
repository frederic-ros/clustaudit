"""
Noise Injector: Inject internal and distant noise
==================================================

This module provides functionality to inject two types of noise:
- Internal noise: ambiguous points between clusters
- Distant noise: outliers far from all clusters

Part of clustaudit.
Date: January 2026 (bridge mode: November 2026)
"""

import numpy as np
import warnings
from scipy.spatial import cKDTree
from ..core.state import ClusteringState
from ..config.params import NoiseParams


def inject_noise(
    state: ClusteringState,
    params: NoiseParams,
    random_state: np.random.RandomState
) -> ClusteringState:
    """
    Inject internal (ambiguous) and distant (outlier) noise.

    Parameters
    ----------
    state : ClusteringState
        Current STATE (cluster cores only)
    params : NoiseParams
        Noise parameters
    random_state : np.random.RandomState
        Random state

    Returns
    -------
    ClusteringState
        Updated STATE with noise points
    """
    if params.noise_ratio == 0:
        return state

    n_current = state.n_samples()
    n_noise_total = int(n_current * params.noise_ratio / (1 - params.noise_ratio))
    n_internal = int(n_noise_total * params.internal_noise_ratio)
    n_distant = n_noise_total - n_internal
    dimension = state.dimension()
    noise_points = []

    # ========================================================================
    # 1. Internal noise (ambiguous points between clusters)
    # ========================================================================
    if n_internal > 0:
        if params.internal_noise_mode == "bridge":
            internal_noise = _bridge_noise(state, n_internal, params, random_state)
        else:  # legacy "bounding_box" (v1): NOT near boundaries in d >= 10
            data_min, data_max = state.Z.min(axis=0), state.Z.max(axis=0)
            data_range = data_max - data_min
            expanded_min = data_min - params.internal_noise_scale * data_range / 2
            expanded_max = data_max + params.internal_noise_scale * data_range / 2
            internal_noise = random_state.uniform(
                expanded_min, expanded_max, size=(n_internal, dimension)
            )
        noise_points.append(internal_noise)

    # ========================================================================
    # 2. Distant noise (outliers)
    # ========================================================================
    if n_distant > 0:
        # Compute cluster centers
        cluster_centers = [
            state.Z[state.y == k].mean(axis=0)
            for k in range(state.n_clusters())
            if np.any(state.y == k)
        ]

        if len(cluster_centers) > 0:
            cluster_centers = np.array(cluster_centers)
            avg_dist = np.mean([
                np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                for i in range(len(cluster_centers))
                for j in range(i + 1, len(cluster_centers))
            ])
        else:
            avg_dist = 1.0

        min_distance = params.distant_noise_factor * avg_dist

        # Different strategies for generating distant noise
        if params.distant_noise_strategy == "rejection_sampling":
            # Build KD-tree once for fast nearest-neighbour queries
            tree = cKDTree(state.Z)
            distant_noise = []
            trials = 0

            while len(distant_noise) < n_distant and trials < params.max_trials:
                # NOTE: the box is 2 x avg_dist beyond the data; in d <= 3 with the
                # default distant_noise_factor = 3 few candidates pass the rejection
                # test (use a smaller factor for 2-D illustrations). Kept as is so that
                # every published run shares one noise model.
                batch = random_state.uniform(
                    state.Z.min(axis=0) - 2 * avg_dist,
                    state.Z.max(axis=0) + 2 * avg_dist,
                    size=(params.batch_size, dimension)
                )

                # Vectorised nearest-neighbour distance for the whole batch
                dists, _ = tree.query(batch, k=1)

                for i in range(len(batch)):
                    if len(distant_noise) >= n_distant:
                        break
                    if dists[i] >= min_distance:
                        distant_noise.append(batch[i])

                trials += params.batch_size

            if len(distant_noise) < n_distant:
                warnings.warn(f"Generated only {len(distant_noise)}/{n_distant} distant noise points")

            if len(distant_noise) > 0:
                noise_points.append(np.array(distant_noise))

        elif params.distant_noise_strategy == "gaussian_far":
            data_center = state.Z.mean(axis=0)
            direction = random_state.normal(size=(n_distant, dimension))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            distant_noise = data_center + direction * min_distance * params.distant_noise_scale
            noise_points.append(distant_noise)

        else:  # uniform_box
            scale = params.distant_noise_factor * params.distant_noise_scale
            data_range = state.Z.max(axis=0) - state.Z.min(axis=0)
            distant_noise = random_state.uniform(
                state.Z.min(axis=0) - scale * data_range,
                state.Z.max(axis=0) + scale * data_range,
                size=(n_distant, dimension)
            )
            noise_points.append(distant_noise)

    # ========================================================================
    # 3. Combine and update state
    # ========================================================================
    if len(noise_points) == 0:
        return state

    all_noise = np.vstack(noise_points)
    n_noise_actual = len(all_noise)
    noise_labels = np.full(n_noise_actual, -1, dtype=int)

    # Update masks
    new_masks = {
        k: np.concatenate([state.masks[k], np.zeros(n_noise_actual, dtype=bool)])
        for k in state.masks
    }
    new_masks['noise'][-n_noise_actual:] = True

    return ClusteringState(
        Z=np.vstack([state.Z, all_noise]),
        y=np.concatenate([state.y, noise_labels]),
        masks=new_masks,
        entropy=np.concatenate([state.entropy, np.zeros(n_noise_actual)]),
        confidence=np.concatenate([state.confidence, np.zeros(n_noise_actual)]),
        metadata=state.metadata.copy(),
        timestamp=state.timestamp,
        random_seed=state.random_seed
    )


def _bridge_noise(
    state: ClusteringState,
    n_internal: int,
    params: NoiseParams,
    random_state: np.random.RandomState
) -> np.ndarray:
    """
    Place `n_internal` points in the regions *between* clusters.

    For each point: pick a random labelled source point x_a of cluster a, find
    its nearest neighbour x_b in any other cluster, draw t ~ U(bridge_t_range)
    and return x_a + t (x_b - x_a) + jitter, where the jitter is isotropic
    Gaussian whose norm is about bridge_jitter * (mean nearest-neighbour
    distance of cluster a). Such points lie where several assignments are plausible, which
    is what "boundary ambiguity" means in the paper. With a single cluster the
    function falls back to jittered copies of random points.
    """
    labelled = state.y >= 0
    Z, y = state.Z[labelled], state.y[labelled]
    labels = np.unique(y)
    n_lab, dim = Z.shape
    if n_lab == 0:
        return np.empty((0, state.dimension()))

    # local scale per cluster = mean 1-NN distance inside the cluster
    local_scale = {}
    trees = {}
    for k in labels:
        Zk = Z[y == k]
        trees[k] = cKDTree(Zk)
        if len(Zk) > 1:
            d1, _ = trees[k].query(Zk, k=2)
            local_scale[k] = float(np.mean(d1[:, 1]))
        else:
            local_scale[k] = 1.0

    # nearest point of *another* cluster, for every labelled point
    nearest_other = np.empty_like(Z)
    for k in labels:
        idx_k = np.where(y == k)[0]
        if len(labels) == 1:
            nearest_other[idx_k] = Z[idx_k]
            continue
        best_d = np.full(len(idx_k), np.inf)
        best_p = np.zeros((len(idx_k), dim))
        for j in labels:
            if j == k:
                continue
            d, i = trees[j].query(Z[idx_k], k=1)
            better = d < best_d
            best_d[better] = d[better]
            best_p[better] = Z[y == j][i[better]]
        nearest_other[idx_k] = best_p

    src = random_state.randint(0, n_lab, size=n_internal)
    t = random_state.uniform(params.bridge_t_range[0], params.bridge_t_range[1],
                             size=(n_internal, 1))
    base = Z[src] + t * (nearest_other[src] - Z[src])
    scales = np.array([local_scale[k] for k in y[src]])[:, None]
    # jitter norm ~ bridge_jitter * local scale (hence the 1/sqrt(dim) per coordinate)
    jitter = random_state.normal(size=(n_internal, dim)) * params.bridge_jitter * scales / np.sqrt(dim)
    return base + jitter
