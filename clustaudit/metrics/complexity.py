# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 12:01:40 2026

@author: frederic.ros
"""

# -*- coding: utf-8 -*-
"""
Clustering Complexity Metrics
-----------------------------

Compute a global clustering difficulty score ∈ [0,1]
based on intrinsic dataset properties.

Author: frederic.ros
"""

import numpy as np


# --------------------------------------------------
# Utils
# --------------------------------------------------

def _normalize(x, min_val=0.0, max_val=1.0):
    """Safe normalization to [0,1]."""
    if np.isnan(x):
        return 0.0
    return np.clip((x - min_val) / (max_val - min_val + 1e-12), 0.0, 1.0)


# --------------------------------------------------
# Main complexity score
# --------------------------------------------------

def compute_complexity_score(metrics, weights=None):
    """
    Compute a global clustering complexity score ∈ [0,1].

    Parameters
    ----------
    metrics : dict
        Output of compute_complete_metrics()
        (typically metrics['global'], ['clusters'], ['boundary'], ['noise'])
    weights : dict, optional
        Custom weights for components

    Returns
    -------
    dict with:
        - complexity_score
        - components (breakdown)
    """

    # Default weights
    if weights is None:
        weights = {
            'separation': 0.30,
            'boundary': 0.30,
            'shape': 0.20,
            'noise': 0.20
        }

    # --------------------------------------------------
    # 1️⃣ Separation difficulty (global)
    # --------------------------------------------------
    silhouette = metrics['global'].get('silhouette', np.nan)
    separation = 1.0 - _normalize(silhouette, min_val=0.0, max_val=0.7)

    # --------------------------------------------------
    # 2️⃣ Boundary complexity
    # --------------------------------------------------
    boundary_ratios = []
    boundary_stds = []

    for b in metrics['boundary'].values():
        if not np.isnan(b['boundary_core_distance_std']):
            boundary_stds.append(b['boundary_core_distance_std'])

    for c in metrics['clusters'].values():
        boundary_ratios.append(c['boundary_ratio'])

    boundary_ratio_mean = np.mean(boundary_ratios) if boundary_ratios else 0.0
    boundary_std_mean = np.mean(boundary_stds) if boundary_stds else 0.0

    boundary_complexity = (
        0.6 * _normalize(boundary_ratio_mean, max_val=0.5) +
        0.4 * _normalize(boundary_std_mean)
    )

    # --------------------------------------------------
    # 3️⃣ Shape / density heterogeneity
    # --------------------------------------------------
    elongations = [
        c['elongation']
        for c in metrics['clusters'].values()
        if not np.isnan(c['elongation'])
    ]

    shape_complexity = (
        np.mean([_normalize(e, max_val=5.0) for e in elongations])
        if elongations else 0.0
    )

    # --------------------------------------------------
    # 4️⃣ Noise difficulty
    # --------------------------------------------------
    noise_ratio = metrics['noise'].get('noise_ratio', 0.0)
    internal_noise_ratio = metrics['noise'].get('internal_noise_ratio', 0.0)

    noise_complexity = (
        0.6 * _normalize(noise_ratio, max_val=0.4) +
        0.4 * _normalize(internal_noise_ratio, max_val=0.3)
    )

    # --------------------------------------------------
    # Final score
    # --------------------------------------------------
    complexity_score = (
        weights['separation'] * separation +
        weights['boundary'] * boundary_complexity +
        weights['shape'] * shape_complexity +
        weights['noise'] * noise_complexity
    )

    return {
        'complexity_score': float(np.clip(complexity_score, 0.0, 1.0)),
        'components': {
            'separation': separation,
            'boundary': boundary_complexity,
            'shape': shape_complexity,
            'noise': noise_complexity
        }
    }
