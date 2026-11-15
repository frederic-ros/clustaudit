"""
Cluster Utilities: Transformations, base distributions, density shaping
=======================================================================

Provides helper functions for cluster generation pipelines.

Part of clustaudit.
Date: February 2026
"""

import numpy as np
from ..config.params import DensityParams, ClusterParams, StructuralParams


def _to_generator(rng):
    """Convert RandomState or int to Generator."""
    if isinstance(rng, np.random.RandomState):
        seed = rng.randint(0, 2**31 - 1, dtype=np.int32)
        return np.random.default_rng(seed)
    elif isinstance(rng, np.random.Generator):
        return rng
    else:
        return np.random.default_rng(rng)


# -------------------------------
# Transformations
# -------------------------------
def rotate_points(points, rng):
    """Random rotation in 2D planes for high-dimensional points."""
    dim = points.shape[1]
    for _ in range(rng.integers(1, max(2, dim // 2))):
        d = rng.choice(dim, size=2, replace=False)
        a = rng.uniform(0, 2 * np.pi)
        x, y = points[:, d[0]].copy(), points[:, d[1]].copy()
        points[:, d[0]] = np.cos(a) * x - np.sin(a) * y
        points[:, d[1]] = np.sin(a) * x + np.cos(a) * y
    return points


def perturb_points(points, rng, perturbation_scale):
    """Apply uniform perturbation inside bounding box."""
    bbox = points.max(axis=0) - points.min(axis=0)
    return points + rng.uniform(-perturbation_scale * bbox,
                                perturbation_scale * bbox,
                                size=points.shape)


# -------------------------------
# Density shaping
# -------------------------------
def apply_intra_density(points, center, dp: DensityParams, rng):
    """
    Intra-cluster density shaping by a monotone radial transform (no resampling,
    no duplicated points). With frac = r / r_max in [0, 1] and s = strength:
      core_dense      : frac -> frac ** (1 + s)        (points pulled toward the centre)
      boundary_dense  : frac -> frac ** (1 / (1 + s))  (points pushed toward the rim)
      radial_gradient : frac -> frac ** (1 + s / 2)    (milder core concentration)
    v1 resampled points with replacement, which duplicated points and was gated
    by `enable_adaptive_density` (False): it never ran.
    """
    if dp.intra_density_mode == "uniform" or dp.intra_density_strength <= 0:
        return points
    s = float(dp.intra_density_strength)
    d = points - center
    r = np.linalg.norm(d, axis=1)
    r_max = r.max() if r.max() > 0 else 1.0
    frac = r / r_max
    if dp.intra_density_mode == "core_dense":
        new = frac ** (1.0 + s)
    elif dp.intra_density_mode == "boundary_dense":
        new = frac ** (1.0 / (1.0 + s))
    else:  # radial_gradient
        new = frac ** (1.0 + s / 2.0)
    scale = np.where(r > 0, new / np.maximum(frac, 1e-12), 1.0)
    return center + d * scale[:, None]


def _sample_inter_density_factor(dp: DensityParams, rng):
    """Sample and clip an inter-cluster density multiplier."""
    if dp.inter_density_distribution == "lognormal":
        factor = rng.lognormal(mean=0, sigma=dp.inter_density_variation)
    else:
        factor = rng.uniform(1 - dp.inter_density_variation, 1 + dp.inter_density_variation)
    return np.clip(factor, *dp.inter_density_clip)


# -------------------------------
# Base distributions
# -------------------------------
def generate_gaussian(center, n_points, cluster_std_range, dp: DensityParams, rng):
    """Generate Gaussian cluster with inter and intra density shaping."""
    std = rng.uniform(*cluster_std_range)
    if dp.inter_density_variation > 0:
        std *= _sample_inter_density_factor(dp, rng)
    points = rng.normal(center, std, size=(n_points, len(center)))
    return apply_intra_density(points, center, dp, rng)


def generate_uniform(center, n_points, cluster_std_range, dp: DensityParams, rng):
    """Generate uniform cluster with inter and intra density shaping."""
    std = rng.uniform(*cluster_std_range)
    if dp.inter_density_variation > 0:
        std *= _sample_inter_density_factor(dp, rng)
    points = rng.uniform(center - std * np.sqrt(3), center + std * np.sqrt(3), size=(n_points, len(center)))
    return apply_intra_density(points, center, dp, rng)


def generate_mixture_gaussian(center, n_points, cluster_std_range, cluster_params, dp: DensityParams, rng):
    """Generate mixture Gaussian or agile cluster with intra density applied last."""
    if cluster_params is not None:
        # generate_agile_cluster is defined below in this same module —
        # no circular import needed.
        points = generate_agile_cluster(center, n_points, cluster_params, rng)
    else:
        std = rng.uniform(*cluster_std_range)
        if dp.inter_density_variation > 0:
            std *= _sample_inter_density_factor(dp, rng)
        n1 = n_points // 2
        offset = rng.normal(0, std, size=len(center))
        points = np.vstack([
            rng.normal(center, std, size=(n1, len(center))),
            rng.normal(center + offset, std, size=(n_points - n1, len(center)))
        ])
    return apply_intra_density(points, center, dp, rng)


def generate_agile_cluster(center: np.ndarray, n_points: int, params: ClusterParams, random_state=None) -> np.ndarray:
    """
    Generate agile cluster with sub-components, cross effects, and anisotropy.

    Note: intra-density shaping is NOT applied here. When called via
    generate_mixture_gaussian, density is applied by the outer wrapper using
    the actual DensityParams. Direct callers who need density shaping should
    apply it explicitly afterwards.
    """
    rng = _to_generator(random_state)
    dim = len(center)

    n_components = rng.integers(params.n_components_range[0], params.n_components_range[1] + 1)
    weights = np.ones(n_components) / n_components if params.weights_type == 'uniform' else np.array([0.5**i for i in range(n_components)])
    if params.weights_type == 'geometric':
        weights /= weights.sum()
    points_list = []

    for k in range(n_components):
        n_k = int(n_points * weights[k])
        sigma_base = rng.uniform(*params.std_range)
        sigma_k = sigma_base * (1 + params.unbalanced_strength * (2 * rng.random() - 1))
        sigma_k = np.clip(sigma_k, params.std_range[0], params.std_range[1])
        sigma_vec = sigma_k * (1 + params.anisotropy_strength * (2 * rng.random(dim) - 1))
        sigma_vec = np.clip(sigma_vec, sigma_k * params.anisotropy_clip[0], sigma_k * params.anisotropy_clip[1])
        cross_strength = rng.uniform(*params.cross_strength_range)
        if cross_strength > 0:
            dominant_dim = rng.integers(0, dim)
            sigma_vec = (1 - cross_strength) * sigma_vec + cross_strength * sigma_k
            sigma_vec[dominant_dim] = sigma_k * (params.cross_clip[0] + (params.cross_clip[1] - params.cross_clip[0]) * cross_strength)
        offset_k = rng.normal(0, sigma_base * params.structure_scale, size=dim)
        max_offset = sigma_base * params.max_offset_factor
        offset_k = np.clip(offset_k, -max_offset, max_offset)
        comp_points = rng.normal(center + offset_k, sigma_vec, size=(n_k, dim))
        points_list.append(comp_points)

    points = np.vstack(points_list)
    while points.shape[0] < n_points:
        extra = rng.normal(center, rng.uniform(*params.std_range), size=(1, dim))
        points = np.vstack([points, extra])
    points = points[:n_points]

    return points


def generate_manifold_cluster(center: np.ndarray, n_points: int, params: StructuralParams, random_state=None) -> np.ndarray:
    """Generate manifold cluster (Swiss Roll or S-Curve) in arbitrary dimensions."""
    rng = _to_generator(random_state)
    center = np.asarray(center)
    D = center.shape[0]
    mp = params.manifold_params
    intrinsic_dim = mp.intrinsic_dim
    manifold_type = mp.manifold_type
    length_scale = mp.length_scale
    thickness = mp.thickness
    orientation_random = mp.orientation_random
    node_radius = mp.node_radius
    complexity_fraction = mp.complexity_fraction
    enable_curvature = mp.enable_local_curvature
    curvature_amp = mp.curvature_amplitude
    max_radius = node_radius * complexity_fraction

    if manifold_type == "swiss_roll":
        t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n_points)
        h = rng.uniform(-1, 1, n_points)
        x = t * np.cos(t)
        y = h
        z = t * np.sin(t)
        intrinsic_points = np.vstack([x, y, z]).T
    elif manifold_type == "s_curve":
        t = rng.uniform(-np.pi, np.pi, n_points)
        h = rng.uniform(-1, 1, n_points)
        x = np.sin(t)
        y = h
        z = np.sign(t) * (1 - np.cos(t))
        intrinsic_points = np.vstack([x, y, z]).T
    else:
        raise ValueError(f"Unknown manifold_type '{manifold_type}'")

    # Scale length
    intrinsic_points[:, :3] *= length_scale

    # Add thickness/noise along non-intrinsic dimensions
    if intrinsic_dim > 1:
        intrinsic_points[:, 1] += rng.normal(0, thickness, size=n_points)
    if intrinsic_dim > 2:
        intrinsic_points[:, 2] += rng.normal(0, thickness, size=n_points)
    
    # Reduce to intrinsic_dim if needed
    intrinsic_points = intrinsic_points[:, :intrinsic_dim]

    # Normalize to max_radius
    norms = np.linalg.norm(intrinsic_points, axis=1)
    max_norm = norms.max()
    if max_norm > 0:
        intrinsic_points *= (max_radius / max_norm)

    # Optional local curvature for S-curve
    if enable_curvature and manifold_type == "s_curve" and intrinsic_dim >= 2:
        t_local = intrinsic_points[:, 0] / max_radius * np.pi
        intrinsic_points[:, 1] += curvature_amp * max_radius * np.sin(3 * t_local)

    # Embed in D dimensions
    if D > intrinsic_dim:
        points = np.zeros((n_points, D))
        points[:, :intrinsic_dim] = intrinsic_points
        if orientation_random:
            A = rng.normal(size=(D, D))
            Q, _ = np.linalg.qr(A)
            points = points @ Q
    else:
        points = intrinsic_points[:, :D]

    # Translate to center
    points += center

    return points

