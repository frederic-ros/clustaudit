"""
clustaudit -- interventional auditing of clustering algorithms
==============================================================

    from clustaudit import generate_dataset, difficulty_profile
    from clustaudit.config import StructuralParams, NoiseParams, AmbiguityParams

    state = generate_dataset(1000, 20, 5, noise=NoiseParams(noise_ratio=0.2), random_seed=42)
    print(profile_from_state(state))
    print(difficulty_profile(X_real, y_real))
"""
__version__ = "2.2.0"

from .core.state import ClusteringState
from .core.pipeline import generate_dataset, generate_easy_dataset, generate_hard_dataset
from .config.params import StructuralParams, NoiseParams, AmbiguityParams, DensityParams, DataGenerationConfig
from .metrics import (difficulty_profile, profile_from_state, DifficultyProfile, knn_profile,
                      compute_metrics_with_and_without_noise, compute_D_dataset)

__all__ = ["ClusteringState", "generate_dataset", "generate_easy_dataset", "generate_hard_dataset",
           "StructuralParams", "NoiseParams", "AmbiguityParams", "DensityParams", "DataGenerationConfig",
           "difficulty_profile", "profile_from_state", "DifficultyProfile", "knn_profile",
           "compute_metrics_with_and_without_noise", "compute_D_dataset", "__version__"]
