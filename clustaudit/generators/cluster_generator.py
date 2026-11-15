"""
Cluster Generator: Generate cluster points with geometric transformations
==========================================================================

Master module orchestrating cluster generation pipelines.

Part of clustaudit.
Date: February 2026
"""

import numpy as np
from ..core.state import ClusteringState
from ..config.params import StructuralParams, DensityParams
from .cluster_utils import (
    _to_generator,
    rotate_points,
    perturb_points,
    generate_gaussian,
    generate_uniform,
    generate_mixture_gaussian,
    generate_agile_cluster,
    generate_manifold_cluster
)


def _generate_cluster_points(
    center: np.ndarray,
    n_points: int,
    params: StructuralParams,
    random_state
) -> np.ndarray:
    """
    Generate points for a single cluster (no state manipulation).

    This is the pure computation core: it advances *random_state* identically
    to one call of the public ``generate_cluster``, so calling it K times in a
    loop produces the same random sequence as K successive ``generate_cluster``
    calls.

    Returns
    -------
    points : np.ndarray, shape (n_points, dim)
    """
    rng = _to_generator(random_state)
    dim = len(center)
    dp: DensityParams = getattr(params, "density_params", DensityParams())

    # -------------------------------
    # Base distribution selection
    # -------------------------------
    dist = params.cluster_distribution.lower()
    if dist == "gaussian":
        points = generate_gaussian(center, n_points, params.cluster_std_range, dp, rng)
    elif dist == "mixture_gaussian":
        points = generate_mixture_gaussian(center, n_points, params.cluster_std_range, params.cluster_params, dp, rng)
    elif dist == "uniform":
        points = generate_uniform(center, n_points, params.cluster_std_range, dp, rng)
    elif dist == "manifold":
        points = generate_manifold_cluster(center, n_points, params, rng)
    else:
        points = generate_uniform(center, n_points, params.cluster_std_range, dp, rng)

    # -------------------------------
    # Geometric transformations
    # -------------------------------
    if params.enable_anisotropy:
        anis_factors = rng.uniform(*params.anisotropy_factor_range, size=dim)
        points *= anis_factors
    if params.enable_rotation and dim >= 2:
        points = rotate_points(points, rng)
    if params.enable_perturbation:
        points = perturb_points(points, rng, params.perturbation_scale)

    # Recentering
    points -= points.mean(axis=0)
    points += center

    return points


def generate_cluster(
    state: ClusteringState,
    center: np.ndarray,
    cluster_label: int,
    n_points: int,
    params: StructuralParams,
    random_state=None
) -> ClusteringState:
    """
    Generate a cluster and append it to the given ClusteringState.

    This function selects the appropriate base distribution, applies
    optional anisotropy, rotation, and perturbation, and updates masks.
    """
    points = _generate_cluster_points(center, n_points, params, random_state)

    # -------------------------------
    # Labels and masks
    # -------------------------------
    labels = np.full(n_points, cluster_label, dtype=int)
    new_masks = {
        k: np.concatenate([
            state.masks[k],
            np.ones(n_points, dtype=bool) if k == 'core' else np.zeros(n_points, dtype=bool)
        ])
        for k in state.masks
    }

    # -------------------------------
    # Update state (metadata copied to prevent shared-mutable-dict bugs)
    # -------------------------------
    return ClusteringState(
        Z=np.vstack([state.Z, points]) if state.n_samples() > 0 else points,
        y=np.concatenate([state.y, labels]) if state.n_samples() > 0 else labels,
        masks=new_masks,
        entropy=np.concatenate([state.entropy, np.zeros(n_points)]),
        confidence=np.concatenate([state.confidence, np.ones(n_points)]),
        metadata=state.metadata.copy(),
        timestamp=state.timestamp,
        random_seed=state.random_seed
    )
