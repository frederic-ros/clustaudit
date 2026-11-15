"""
Master Generator: Main orchestrator for synthetic dataset generation
=====================================================================

This module implements Algorithm 1 of the paper,
orchestrating the complete pipeline for synthetic dataset generation.

Part of clustaudit.
Date: February 2026
"""

import numpy as np
from datetime import datetime
from typing import Optional, Union, Tuple
from ..core.state import ClusteringState
from ..config.params import (
    StructuralParams,
    NoiseParams,
    AmbiguityParams,
    DataGenerationConfig
)
from ..generators.hypercube import sample_hypercube_centers
from ..generators.cluster_generator import _generate_cluster_points
from ..generators.noise import inject_noise
from ..generators.refinement import refine_labels
from ..utils.tracing import GenerationTrace


def generate_dataset(
    n_samples: int,
    dimension: int,
    n_clusters: int,
    structural: Optional[StructuralParams] = None,
    noise: Optional[NoiseParams] = None,
    ambiguity: Optional[AmbiguityParams] = None,
    random_seed: Optional[int] = None,
    pipeline_steps: Optional[dict] = None,
    record_trace: bool = False
) -> Union[ClusteringState, Tuple[ClusteringState, GenerationTrace]]:
    """
    Master synthetic dataset generation pipeline with step control.
    """
    # --------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------
    config = DataGenerationConfig(
        n_samples=n_samples,
        dimension=dimension,
        n_clusters=n_clusters,
        random_seed=random_seed,
        structural=structural or StructuralParams(),
        noise=noise or NoiseParams(),
        ambiguity=ambiguity or AmbiguityParams()
    )
    rng = np.random.RandomState(random_seed)
    trace = GenerationTrace(steps=[]) if record_trace else None

    # Pipeline defaults
    steps = {
        'hypercube': True,
        'generate_clusters': True,
        'pre_refine': True,
        'inject_noise': True,
        'post_refine': True
    }
    if pipeline_steps:
        steps.update(pipeline_steps)

    # --------------------------------------------------------------------
    # Step 1: Initialize empty state
    # --------------------------------------------------------------------
    state = ClusteringState.empty(random_seed=random_seed)
    sizes = None  # declared here so metadata update is always safe

    # --------------------------------------------------------------------
    # Step 2: Sample cluster centers on hypercube
    # --------------------------------------------------------------------
    if steps['hypercube']:
        centers = sample_hypercube_centers(
            dimension,
            n_clusters,
            config.structural.hypercube_connectivity,
            config.structural.min_hamming_ratio,
            rng
        )
    else:
        centers = np.zeros((n_clusters, dimension))

    # --------------------------------------------------------------------
    # Step 3: Generate cluster points (batch assembly)
    # --------------------------------------------------------------------
    if steps['generate_clusters']:
        n_core = int(n_samples * (1 - config.noise.noise_ratio))

        # Compute cluster sizes
        if config.structural.cluster_size_imbalance == 1.0:
            sizes = [n_core // n_clusters] * n_clusters
            for i in range(n_core % n_clusters):
                sizes[i] += 1
        else:
            weights = np.exp(-np.arange(n_clusters) * np.log(config.structural.cluster_size_imbalance))
            weights /= weights.sum()
            sizes = (weights * n_core).astype(int)
            sizes = np.maximum(sizes, config.structural.min_cluster_size)
            diff = n_core - sizes.sum()
            sizes[0] += diff

        # Generate all cluster points — each call to _generate_cluster_points
        # advances `rng` by exactly one int (same sequence as the old loop
        # with generate_cluster), so reproducibility is preserved.
        all_points = []
        all_labels = []
        for k in range(n_clusters):
            points = _generate_cluster_points(centers[k], sizes[k], config.structural, rng)
            all_points.append(points)
            all_labels.append(np.full(sizes[k], k, dtype=int))

        # Assemble state in a single pass (avoids O(K²) repeated vstacks)
        n_core_total = sum(sizes)
        state = ClusteringState(
            Z=np.vstack(all_points),
            y=np.concatenate(all_labels),
            masks={
                'core':          np.ones(n_core_total, dtype=bool),
                'boundary':      np.zeros(n_core_total, dtype=bool),
                'noise':         np.zeros(n_core_total, dtype=bool),
                'internal_noise': np.zeros(n_core_total, dtype=bool),
                'distant_noise': np.zeros(n_core_total, dtype=bool),
            },
            entropy=np.zeros(n_core_total),
            confidence=np.ones(n_core_total),
            metadata={},
            timestamp=datetime.now(),
            random_seed=random_seed
        )

        if trace:
            trace.log("clusters_generated", state, info={"centers": centers, "cluster_sizes": sizes})

    # --------------------------------------------------------------------
    # Step 4: Pre-noise label refinement
    # --------------------------------------------------------------------
    if steps['pre_refine']:
        state = refine_labels(
            state,
            config.ambiguity.pre_refinement_classifier,
            config.ambiguity,
            "pre-noise",
            rng
        )
        if trace:
            trace.log("pre_refinement", state)

    # --------------------------------------------------------------------
    # Step 5: Inject noise
    # --------------------------------------------------------------------
    if steps['inject_noise']:
        state = inject_noise(state, config.noise, rng)
        if trace:
            trace.log("noise_injected", state)

    # --------------------------------------------------------------------
    # Step 6: Post-noise label refinement
    # --------------------------------------------------------------------
    if steps['post_refine']:
        state = refine_labels(
            state,
            config.ambiguity.post_refinement_classifier,
            config.ambiguity,
            "post-noise",
            rng
        )
        if trace:
            trace.log("post_refinement", state)

    # --------------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------------
    state.metadata.update({
        'centers': centers,
        'cluster_sizes': sizes,
        'n_clusters': n_clusters,
        'dimension': dimension,
        'noise_ratio': config.noise.noise_ratio,
        'random_seed': random_seed,
        'generation_params': config
    })

    if trace:
        return state, trace
    return state


def generate_easy_dataset(
    n_samples: int = 1000,
    dimension: int = 10,
    random_seed: int = 42
) -> ClusteringState:
    """
    Generate easy dataset (well-separated, no noise).
    """
    return generate_dataset(
        n_samples, dimension, 4,
        structural=StructuralParams(
            enable_anisotropy=False,
            enable_rotation=False,
            enable_perturbation=False,
            cluster_std_range=(0.1, 0.1),
            hypercube_connectivity=0.8
        ),
        noise=NoiseParams(noise_ratio=0.0),
        ambiguity=AmbiguityParams(
            enable_pre_refinement=False,
            enable_post_refinement=False
        ),
        random_seed=random_seed
    )


def generate_hard_dataset(
    n_samples: int = 1000,
    dimension: int = 50,
    random_seed: int = 42
) -> ClusteringState:
    """
    Generate hard dataset (high-dim, noisy, imbalanced, anisotropic).
    """
    return generate_dataset(
        n_samples, dimension, 8,
        structural=StructuralParams(
            hypercube_connectivity=0.3,
            anisotropy_factor_range=(2.0, 5.0),
            enable_rotation=True,
            enable_perturbation=True,
            cluster_size_imbalance=3.0,
            min_cluster_size=20
        ),
        noise=NoiseParams(
            noise_ratio=0.20,
            internal_noise_ratio=0.6,
            distant_noise_factor=4.0
        ),
        ambiguity=AmbiguityParams(),
        random_seed=random_seed
    )
