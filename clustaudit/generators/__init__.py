"""Generators for synthetic clustering components."""

# Core cluster generators (master file)
from .cluster_generator import (
    generate_cluster,
    generate_agile_cluster,
    generate_manifold_cluster
)

# Utilities and transformations (optional to expose)
from .cluster_utils import (
    rotate_points,
    perturb_points,
    apply_intra_density,
    generate_gaussian,
    generate_uniform,
    generate_mixture_gaussian
)

# Other modules
from .hypercube import sample_hypercube_centers
from .noise import inject_noise
from .refinement import refine_labels

__all__ = [
    # Core cluster generation
    'generate_cluster',
    'generate_agile_cluster',
    'generate_manifold_cluster',

    # Utilities (optional, useful for advanced users or testing)
    'rotate_points',
    'perturb_points',
    'apply_intra_density',
    'generate_gaussian',
    'generate_uniform',
    'generate_mixture_gaussian',

    # Other components
    'sample_hypercube_centers',
    'inject_noise',
    'refine_labels'
]
