"""
Dataset Generator
=================

Generate clustering datasets with different intrinsic difficulty levels
and save them as .npz files.

Each .npz contains:
    - X : (n, d) float array
    - y : (n,) integer labels
    - difficulty_param : intrinsic difficulty (D_dataset)
    - seed : random seed

Author: R. Riad
"""

import copy
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from clustaudit.core.pipeline import generate_dataset
from clustaudit.config.params import (
    StructuralParams,
    NoiseParams,
    AmbiguityParams,
    DensityParams,
)
from clustaudit.metrics import profile_from_state


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default output: <cwd>/datasets/suite (override with generate_all(out_dir=...))
DEFAULT_OUTPUT_DIR = Path("datasets") / "suite"

SEEDS = [42, 123, 456, 789, 1011]

N_SAMPLES = 1000
DIMENSION = 20
N_CLUSTERS = 5


# =============================================================================
# DIFFICULTY CONFIGURATIONS
# =============================================================================

def create_difficulty_configs():

    configs = []

    base_configs = {

        "easy": {
            "structural": StructuralParams(
                hypercube_connectivity=0.8,
                enable_anisotropy=False,
                enable_rotation=False,
                cluster_std_range=(0.1, 0.1),
                cluster_size_imbalance=1.0,
                density_params=DensityParams(
                    inter_density_variation=0.0,
                    intra_density_strength=0.0,
                ),
            ),
            "noise": NoiseParams(noise_ratio=0.0),
            # Theta_amb is the *measurement* instrument: it is identical in every
            # regime (v1 disabled it on 'easy' and used core_threshold=0.90 on 'hard',
            # which made the regimes incommensurable).
            "ambiguity": AmbiguityParams(),
        },

        "medium": {
            "structural": StructuralParams(
                hypercube_connectivity=0.5,
                enable_anisotropy=True,
                anisotropy_factor_range=(1.5, 2.5),
                enable_rotation=True,
                cluster_std_range=(0.15, 0.25),
                cluster_size_imbalance=1.3,
                # density heterogeneity is audited on its own sweep axes, not in the suite
                density_params=DensityParams(inter_density_variation=0.0, intra_density_strength=0.0),
            ),
            "noise": NoiseParams(noise_ratio=0.10),
            "ambiguity": AmbiguityParams(
                enable_pre_refinement=True,
                enable_post_refinement=True,
            ),
        },

        "hard": {
            "structural": StructuralParams(
                hypercube_connectivity=0.3,
                enable_anisotropy=True,
                anisotropy_factor_range=(2.0, 3.0),
                enable_rotation=True,
                enable_perturbation=True,
                cluster_std_range=(0.4, 0.6),   # v2.1.2: real inter-cluster overlap in the hard regime (was (0.2, 0.3): no overlap, kNN error < 1%)
                cluster_size_imbalance=1.5,
                density_params=DensityParams(inter_density_variation=0.0, intra_density_strength=0.0),
            ),
            "noise": NoiseParams(noise_ratio=0.15),
            "ambiguity": AmbiguityParams(
                enable_pre_refinement=True,
                enable_post_refinement=True,
            ),
        },
    }

    config_id = 0

    for difficulty in ["easy", "medium", "hard"]:

        for variant in range(5):

            structural = copy.deepcopy(base_configs[difficulty]["structural"])
            noise = copy.deepcopy(base_configs[difficulty]["noise"])

            if structural.enable_anisotropy and structural.anisotropy_factor_range:

                amin, amax = structural.anisotropy_factor_range

                structural.anisotropy_factor_range = (
                    min(amin + 0.5 * variant, amax),
                    min(amax + 0.5 * variant, amax + 2.5),
                )

            noise.noise_ratio = min(
                noise.noise_ratio + 0.05 * variant,
                0.30,
            )

            configs.append(
                {
                    "config_id": config_id,
                    "difficulty": difficulty,
                    "variant": variant,
                    "structural": structural,
                    "noise": noise,
                    "ambiguity": base_configs[difficulty]["ambiguity"],
                }
            )

            config_id += 1

    return configs


# =============================================================================
# GENERATE DATASETS
# =============================================================================

def generate_all(out_dir=None, seeds=None, n_samples=None, dimension=None, n_clusters=None):
    """Generate the 15 configs x 5 seeds paper suite. Returns the manifest DataFrame."""
    OUTPUT_DIR = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEEDS_ = seeds or SEEDS
    N_ = n_samples or N_SAMPLES
    D_ = dimension or DIMENSION
    K_ = n_clusters or N_CLUSTERS

    configs = create_difficulty_configs()

    manifest = []

    dataset_id = 0

    for config in configs:

        for seed in SEEDS_:

            print(
                f"Generating dataset {dataset_id:03d} "
                f"({config['difficulty']}, variant={config['variant']}), "
                f"seed={seed}"
            )

            state = generate_dataset(
                n_samples=N_,
                dimension=D_,
                n_clusters=K_,
                structural=config["structural"],
                noise=config["noise"],
                ambiguity=config["ambiguity"],
                random_seed=seed,
            )

            X = state.Z.astype(np.float64)
            y = state.y.astype(np.int32)

            prof = profile_from_state(state, seed=seed)
            D_dataset = prof.score

            filename = (
                f"dataset_{dataset_id:03d}_"
                f"{config['difficulty']}_"
                f"v{config['variant']}_"
                f"s{seed}.npz"
            )

            np.savez_compressed(
                OUTPUT_DIR / filename,
                X=X,
                y=y,
                difficulty_param=np.float32(D_dataset),
                seed=np.int32(seed),
            )

            manifest.append(
                {
                    "dataset": filename,
                    "difficulty": config["difficulty"],
                    "variant": config["variant"],
                    "difficulty_param": D_dataset,
                    "seed": seed,
                    "n_samples": X.shape[0],
                    "dimension": X.shape[1],
                    "n_clusters": len(np.unique(y[y >= 0])),
                    **{k: v for k, v in prof.to_dict().items()
                       if k not in ("D_dataset", "n_samples", "dimension", "n_clusters")},
                }
            )

            dataset_id += 1

    df = pd.DataFrame(manifest)
    df.to_csv(OUTPUT_DIR / "manifest.csv", index=False)

    print()
    print(f"Generated {dataset_id} datasets.")
    print(f"Saved to: {OUTPUT_DIR}")
    return df


# =============================================================================

if __name__ == "__main__":
    generate_all()