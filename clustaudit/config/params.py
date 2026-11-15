"""
Parameter Classes for Synthetic Clustering Dataset Generation
==============================================================

This module defines parameter sets (Θ) for controlling the generation
of synthetic clustering datasets.

Part of clustaudit.
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, Dict
@dataclass

class DensityParams:
    """
    Parameters controlling intra- and inter-cluster density variations.

    Density is treated as a structural property of clusters, modulating
    point sampling without altering cluster geometry or noise injection.
    """

    # -------------------------
    # Inter-cluster density
    # -------------------------
    inter_density_variation: float = 0.0
    """
    Strength of density variation between clusters (>= 0).

    0.0 = all clusters have comparable densities
    >0  = clusters have heterogeneous compactness
    """

    inter_density_distribution: Literal["lognormal", "uniform"] = "lognormal"
    """
    Distribution used to sample relative cluster densities.
    """

    inter_density_clip: Tuple[float, float] = (0.5, 2.0)
    """
    Clipping range for relative density multipliers between clusters.
    """

    # -------------------------
    # Intra-cluster density
    # -------------------------
    intra_density_mode: Literal[
        "uniform",
        "core_dense",
        "boundary_dense",
        "radial_gradient"
    ] = "uniform"
    """
    Type of intra-cluster density deformation.

    - uniform: standard i.i.d. sampling
    - core_dense: higher density near cluster center
    - boundary_dense: higher density near cluster boundary
    - radial_gradient: continuous density gradient
    """

    intra_density_strength: float = 0.0
    """
    Strength of intra-cluster density deformation (>= 0).

    0.0 = no deformation (uniform)
    Higher values increase density contrast.
    """

    intra_density_radius_fraction: float = 0.5
    """
    Radius fraction defining the dense core or boundary region (0 to 1).
    """

    enable_adaptive_density: bool = True
    """[deprecated, no effect] v1 gated both density mechanisms behind this flag,
    whose default was False, so neither ever ran. Kept for config compatibility."""

    def validate(self) -> None:
        """Validate density parameter constraints."""
        assert self.inter_density_variation >= 0.0, \
            "inter_density_variation must be >= 0"
        assert 0.0 < self.inter_density_clip[0] <= self.inter_density_clip[1], \
            "Invalid inter_density_clip range"
        assert self.intra_density_strength >= 0.0, \
            "intra_density_strength must be >= 0"
        assert 0.0 < self.intra_density_radius_fraction <= 1.0, \
            "intra_density_radius_fraction must be in (0, 1]"


@dataclass
class ManifoldParams:
    """Parameters for nonlinear manifold-based cluster generation."""
    
    manifold_type: Literal["swiss_roll", "s_curve"] = "swiss_roll"
    """Type of manifold structure."""
    
    intrinsic_dim: int = 2
    """Intrinsic dimension of the manifold."""
    
    length_scale: float = 1.0
    """Length scale of the manifold."""
    
    thickness: float = 0.05
    """Thickness of the manifold (orthogonal noise)."""
    
    orientation_random: bool = True
    """Apply random global rotation if D > intrinsic_dim."""
    
    node_radius: float = 0.5
    """Radius of the cluster around the node."""
    
    complexity_fraction: float = 0.8
    """Fraction for boundary complexity."""
    
    enable_local_curvature: bool = False
    """Enable local curvature effects."""
    
    curvature_amplitude: float = 0.05
    """Amplitude of local curvature (fraction of length_scale)."""
    
    def validate(self) -> None:
        """Validate parameter constraints."""
        assert self.manifold_type in {"swiss_roll", "s_curve"}, \
            f"manifold_type must be 'swiss_roll' or 's_curve', got {self.manifold_type}"
        assert self.intrinsic_dim >= 2, "intrinsic_dim must be >= 2"
        assert self.length_scale > 0, "length_scale must be positive"
        assert self.thickness >= 0, "thickness must be non-negative"
        assert self.node_radius > 0, "node_radius must be positive"
        assert 0 <= self.complexity_fraction <= 1, "complexity_fraction must be in [0, 1]"
        assert 0 <= self.curvature_amplitude <= 1, "curvature_amplitude must be a fraction (0-1)"


@dataclass
class ClusterParams:
    """Parameters for agile cluster generation with cross effects."""
    
    n_components_range: Tuple[int, int] = (1, 3)
    """Number of sub-blobs per cluster (min, max)."""
    
    std_range: Tuple[float, float] = (0.05, 0.25)
    """Standard deviation range for overlap control (min, max)."""
    
    structure_scale: float = 1.0
    """Maximum scale of component offsets relative to center."""
    
    unbalanced_strength: float = 0.3
    """Variation of σ between components. 0 = uniform, 1 = max variation."""
    
    anisotropy_strength: float = 0.3
    """Variation of σ per dimension. 0 = isotropic, 1 = strong anisotropy."""
    
    anisotropy_clip: Tuple[float, float] = (0.5, 1.5)
    """Clip σ per dimension after anisotropy (min, max multiplier)."""
    
    max_offset_factor: float = 1.0
    """Limit max offset of a component relative to σ."""
    
    cross_strength_range: Tuple[float, float] = (0.0, 0.8)
    """Control "cross" effect (dominant dimension). 0 = no effect, 1 = max."""
    
    cross_clip: Tuple[float, float] = (0.5, 1.5)
    """Clip for dominant component in cross effect."""
    
    weights_type: Literal['uniform', 'geometric'] = 'uniform'
    """'uniform' = equal weights, 'geometric' = decreasing."""
    
    extra_std_scale: float = 1.0
    """Multiplier for extra points if needed."""

    def validate(self) -> None:
        """Validate cluster parameter constraints."""
        # Components
        assert self.n_components_range[0] >= 1, \
            "n_components_range min must be >= 1"
        assert self.n_components_range[1] >= self.n_components_range[0], \
            "n_components_range max must be >= min"

        # Standard deviations
        assert self.std_range[0] > 0, \
            "std_range min must be > 0"
        assert self.std_range[1] >= self.std_range[0], \
            "std_range max must be >= min"

        # Structural scales
        assert self.structure_scale > 0, \
            "structure_scale must be positive"
        assert self.max_offset_factor >= 0, \
            "max_offset_factor must be >= 0"
        assert self.extra_std_scale > 0, \
            "extra_std_scale must be positive"

        # Strength parameters
        assert 0.0 <= self.unbalanced_strength <= 1.0, \
            "unbalanced_strength must be in [0, 1]"
        assert 0.0 <= self.anisotropy_strength <= 1.0, \
            "anisotropy_strength must be in [0, 1]"

        # Clips
        assert 0.0 < self.anisotropy_clip[0] <= self.anisotropy_clip[1], \
            "Invalid anisotropy_clip range"
        assert 0.0 <= self.cross_strength_range[0] <= self.cross_strength_range[1] <= 1.0, \
            "cross_strength_range must be in [0, 1]"
        assert 0.0 < self.cross_clip[0] <= self.cross_clip[1], \
            "Invalid cross_clip range"

        # Weights
        assert self.weights_type in {"uniform", "geometric"}, \
            f"Invalid weights_type: {self.weights_type}"

    
@dataclass
class StructuralParams:
    """
    Θ_struct: Structural parameters controlling cluster geometry and separability.

    These parameters define the fundamental geometric structure of clusters,
    including their distribution, transformations, spatial relationships,
    and density properties.
    """

    # Hypercube connectivity
    hypercube_connectivity: float = 0.5
    min_hamming_ratio: float = 0.3

    # Cluster point distribution
    cluster_distribution: Literal["gaussian", "mixture_gaussian", "uniform", "manifold"] = "gaussian"
    cluster_std_range: Tuple[float, float] = (0.1, 0.3)

    # Density modulation (NEW)
    density_params: DensityParams = field(default_factory=DensityParams)
    """Intra- and inter-cluster density modulation parameters."""

    # Geometric transformations
    enable_anisotropy: bool = True
    anisotropy_factor_range: Tuple[float, float] = (1.5, 4.0)
    enable_rotation: bool = True
    enable_nonlinear_deformation: bool = False

    enable_perturbation: bool = True
    perturbation_scale: float = 0.1

    # Cluster size constraints
    min_cluster_size: int = 10
    cluster_size_imbalance: float = 1.0

    # Sub-parameters
    cluster_params: ClusterParams = field(default_factory=ClusterParams)
    manifold_params: ManifoldParams = field(default_factory=ManifoldParams)

    def validate(self) -> None:
        assert 0.0 <= self.hypercube_connectivity <= 1.0
        assert 0.0 <= self.min_hamming_ratio <= 1.0
        assert self.cluster_std_range[0] > 0
        assert self.cluster_std_range[1] >= self.cluster_std_range[0]
        assert self.anisotropy_factor_range[0] >= 1.0
        assert self.anisotropy_factor_range[1] >= self.anisotropy_factor_range[0]
        assert 0.0 <= self.perturbation_scale <= 1.0
        assert self.min_cluster_size > 0
        assert self.cluster_size_imbalance >= 1.0

        self.cluster_params.validate()
        self.manifold_params.validate()
        self.density_params.validate()

@dataclass
class NoiseParams:
    """
    Θ_noise: Noise parameters controlling ambiguity and outliers.
    
    These parameters define two types of noise:
    - Internal noise: points in ambiguous regions between clusters
    - Distant noise: outliers far from all clusters
    """
    
    # Overall noise control
    noise_ratio: float = 0.1
    """Total fraction of points that are noise (0 to 1)."""
    
    internal_noise_ratio: float = 0.7
    """Fraction of noise that is internal/ambiguous (0 to 1)."""
    
    # Internal noise (ambiguous points)
    internal_noise_mode: Literal["bridge", "bounding_box"] = "bridge"
    """How internal (ambiguous) noise points are placed.

    - "bridge"       : each point is drawn on the segment joining a point of one
                       cluster to its nearest point in *another* cluster, plus a
                       jitter proportional to the local scale. The point therefore
                       lies in the inter-cluster region where assignment is ambiguous.
                       This is the behaviour the paper describes.
    - "bounding_box" : legacy behaviour (v1): uniform sampling over the bounding
                       box of the data enlarged by `internal_noise_scale`. In d >= 10
                       such points are far from every cluster and are *not*
                       ambiguous. Kept for reproducibility of the v1 experiments only.
    """

    bridge_t_range: Tuple[float, float] = (0.25, 0.75)
    """Position on the inter-cluster segment (0 = source cluster, 1 = target cluster)."""

    bridge_jitter: float = 0.5
    """Isotropic jitter added to a bridge point, as a multiple of the local scale
    (mean nearest-neighbour distance of the source cluster)."""

    internal_noise_scale: float = 1.5
    """[legacy, bounding_box mode only] Enlargement of the bounding box."""
    
    internal_noise_classification_threshold: float = 0.8
    """KNN probability threshold for classifying internal noise (0 to 1)."""
    
    # Distant noise (outliers)
    distant_noise_factor: float = 3.0
    """Minimum distance factor for outliers (× average intra-cluster distance)."""
    
    distant_noise_scale: float = 1.0
    """Additional scaling for outlier distance."""
    
    distant_noise_strategy: Literal["rejection_sampling", "gaussian_far", "uniform_box"] = "rejection_sampling"
    """Strategy for generating distant noise points."""
    
    # Sampling control
    batch_size: int = 100
    """Batch size for rejection sampling."""
    
    max_trials: int = 10000
    """Maximum trials for rejection sampling before fallback."""
    
    enable_adaptive_factor: bool = True
    """Whether to adaptively adjust distant_noise_factor if sampling fails."""
    
    def validate(self) -> None:
        """Validate parameter constraints."""
        assert 0.0 <= self.noise_ratio <= 1.0, "noise_ratio must be in [0, 1]"
        assert 0.0 <= self.internal_noise_ratio <= 1.0, "internal_noise_ratio must be in [0, 1]"
        assert self.internal_noise_scale > 0, "internal_noise_scale must be positive"
        assert 0.0 <= self.bridge_t_range[0] <= self.bridge_t_range[1] <= 1.0, "bridge_t_range in [0, 1]"
        assert self.bridge_jitter >= 0, "bridge_jitter must be >= 0"
        assert 0.0 <= self.internal_noise_classification_threshold <= 1.0, "threshold in [0, 1]"
        assert self.distant_noise_factor > 0, "distant_noise_factor must be positive"
        assert self.distant_noise_scale > 0, "distant_noise_scale must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_trials > 0, "max_trials must be positive"


@dataclass
class AmbiguityParams:
    """
    Θ_amb: Ambiguity parameters controlling label refinement and entropy computation.
    
    These parameters define how ambiguity is measured and how labels are refined
    to achieve high consistency before noise injection.
    """
    
    # Pre-refinement (before noise injection)
    enable_pre_refinement: bool = True
    """Whether to refine labels before adding noise."""
    
    pre_refinement_classifier: Literal["knn", "svm", "random_forest"] = "knn"
    """Classifier used for pre-refinement."""
    
    pre_refinement_threshold: float = 0.999
    """Convergence threshold for pre-refinement (accuracy)."""
    
    pre_refinement_k_neighbors: Optional[int] = None
    """Number of neighbors for KNN (None = auto: sqrt(n_samples))."""
    
    # Post-refinement (after noise injection)
    enable_post_refinement: bool = True
    """Whether to refine labels after adding noise."""
    
    post_refinement_classifier: Literal["knn", "svm", "random_forest"] = "knn"
    """Classifier used for post-refinement (typically same as pre)."""
    
    confidence_mode: Literal["cv", "in_sample"] = "cv"
    """How confidence/entropy are obtained for the labelled (non-noise) points.

    - "cv"        : out-of-sample, by K-fold cross-validated predicted probabilities
                    (sklearn.model_selection.cross_val_predict). Recommended: an
                    in-sample Random Forest is ~100% confident on every training
                    point, which would classify everything as core.
    - "in_sample" : legacy v1 behaviour (fit on all points, predict on the same).
    Noise points (y == -1) are never in the training set, so they are always
    scored out-of-sample.
    """

    cv_folds: int = 5
    """Number of folds when confidence_mode == "cv"."""

    # Ambiguity categorization thresholds
    core_threshold: float = 0.95
    """Confidence threshold for core points (high confidence)."""
    
    noise_threshold: float = 0.05
    """[legacy] Confidence threshold used in v1 to split noise into internal/distant.
    NOTE: a K-class classifier never returns max-probability below 1/K, so with
    K <= 20 this threshold could never be reached and the `distant_noise` mask was
    always empty in v1. The internal/distant split is now geometric (see
    `distant_noise_quantile`); this field is kept only for the v1 reproduction path.
    """

    distant_noise_quantile: float = 0.10
    """Gap-scale quantile for the internal/distant split of noise points.
    Gap scale = this quantile of {distance from a labelled point to the nearest
    point of another cluster} (a low quantile = the points that face a gap). A
    noise point whose distance to the nearest labelled point exceeds the gap
    scale is *distant*; otherwise it is *internal* (inside or between supports,
    where assignment is ambiguous)."""
    
    # Entropy computation
    entropy_method: Literal["shannon", "gini", "renyi"] = "shannon"
    """Method for computing pointwise entropy e(x_i)."""
    
    entropy_normalization: bool = True
    """Whether to normalize entropy to [0, 1] range."""
    
    def validate(self) -> None:
        """Validate parameter constraints."""
        assert 0.0 <= self.pre_refinement_threshold <= 1.0, "pre_refinement_threshold in [0, 1]"
        if self.pre_refinement_k_neighbors is not None:
            assert self.pre_refinement_k_neighbors > 0, "k_neighbors must be positive"
        assert 0.0 <= self.core_threshold <= 1.0, "core_threshold must be in [0, 1]"
        assert 0.0 <= self.noise_threshold <= 1.0, "noise_threshold must be in [0, 1]"
        assert self.noise_threshold < self.core_threshold, "noise_threshold must be < core_threshold"
        assert 0.0 < self.distant_noise_quantile <= 1.0, "distant_noise_quantile in (0, 1]"
        assert self.cv_folds >= 2, "cv_folds must be >= 2"


@dataclass
class DataGenerationConfig:
    """
    Complete configuration for synthetic clustering dataset generation.
    
    Combines basic dataset parameters with the three parameter sets (Θ).
    """
    
    # Basic dataset parameters
    n_samples: int
    """Total number of points to generate."""
    
    dimension: int
    """Dimensionality of the feature space."""
    
    n_clusters: int
    """Number of clusters to generate."""
    
    random_seed: Optional[int] = None
    """Random seed for reproducibility (None = non-deterministic)."""
    
    # Three parameter sets (Θ)
    structural: StructuralParams = field(default_factory=StructuralParams)
    """Θ_struct: Structural parameters."""
    
    noise: NoiseParams = field(default_factory=NoiseParams)
    """Θ_noise: Noise parameters."""
    
    ambiguity: AmbiguityParams = field(default_factory=AmbiguityParams)
    """Θ_amb: Ambiguity parameters."""
    
    # Pipeline control
    pipeline_steps: Dict[str, bool] = field(default_factory=lambda: {
        'hypercube': True,
        'generate_clusters': True,
        'pre_refine': True,
        'inject_noise': True,
        'post_refine': True
    })
    """Control which pipeline steps to execute."""
    
    def validate(self) -> None:
        """Validate all parameters and their interactions."""
        # Validate basic parameters
        assert self.n_samples > 0, "n_samples must be positive"
        assert self.dimension > 0, "dimension must be positive"
        assert self.n_clusters > 0, "n_clusters must be positive"
        assert self.n_clusters <= self.n_samples, "n_clusters must be <= n_samples"
        
        # Validate minimum cluster size constraint
        n_core_points = int(self.n_samples * (1 - self.noise.noise_ratio))
        min_points_per_cluster = n_core_points // self.n_clusters
        assert min_points_per_cluster >= self.structural.min_cluster_size, \
            f"Too many clusters or too much noise: only {min_points_per_cluster} points per cluster available"
        
        # Validate each parameter set
        self.structural.validate()
        self.noise.validate()
        self.ambiguity.validate()
        
        # Cross-parameter validation
        if self.ambiguity.pre_refinement_k_neighbors is not None:
            assert self.ambiguity.pre_refinement_k_neighbors < n_core_points, \
                "k_neighbors must be < number of core points"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
