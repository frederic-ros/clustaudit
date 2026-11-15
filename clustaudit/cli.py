"""
clustaudit command line
=======================

    clustaudit generate --preset hard --n 1000 --d 20 --k 5 --seed 42 --out datasets/
    clustaudit generate --config configs/hard.yaml
    clustaudit suite    --out datasets/suite             # the 75-dataset paper suite
    clustaudit profile  datasets/suite/dataset_000_easy_v1_s42.npz
    clustaudit profile  my_data.csv --label-col target   # any labelled CSV
    clustaudit profile  --sklearn iris                   # a bundled sklearn dataset

`profile` prints the difficulty profile of a labelled dataset: the four
components, the k-NN descriptors, the simple baselines and the v1 scalar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config.params import (StructuralParams, NoiseParams, AmbiguityParams,
                            DensityParams, DataGenerationConfig)
from .core.pipeline import generate_dataset
from .metrics import profile_from_state, difficulty_profile
from .io.config_io import load_raw, config_from_dict, save_effective_config


# ---------------------------------------------------------------------------
# Presets (the three regimes of the paper)
# ---------------------------------------------------------------------------
def preset_config(preset: str, n: int, d: int, k: int, seed: int) -> DataGenerationConfig:
    presets = {
        "easy": dict(
            structural=StructuralParams(hypercube_connectivity=0.8, enable_anisotropy=False,
                                        enable_rotation=False, cluster_std_range=(0.1, 0.1),
                                        cluster_size_imbalance=1.0,
                                        density_params=DensityParams(0.0, intra_density_strength=0.0)),
            noise=NoiseParams(noise_ratio=0.0),
            ambiguity=AmbiguityParams(),
        ),
        "medium": dict(
            structural=StructuralParams(hypercube_connectivity=0.5, enable_anisotropy=True,
                                        anisotropy_factor_range=(1.5, 2.5), enable_rotation=True,
                                        cluster_std_range=(0.15, 0.25), cluster_size_imbalance=1.3,
                                        density_params=DensityParams(0.3, intra_density_strength=0.5)),
            noise=NoiseParams(noise_ratio=0.10),
            ambiguity=AmbiguityParams(),
        ),
        "hard": dict(
            structural=StructuralParams(hypercube_connectivity=0.3, enable_anisotropy=True,
                                        anisotropy_factor_range=(2.0, 3.0), enable_rotation=True,
                                        enable_perturbation=True, cluster_std_range=(0.2, 0.3),
                                        cluster_size_imbalance=1.5,
                                        density_params=DensityParams(0.5, intra_density_strength=1.0)),
            noise=NoiseParams(noise_ratio=0.15),
            ambiguity=AmbiguityParams(),
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}' (easy/medium/hard)")
    p = presets[preset]
    return DataGenerationConfig(n_samples=n, dimension=d, n_clusters=k, random_seed=seed, **p)


def generate_and_save(config: DataGenerationConfig, name: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = generate_dataset(config.n_samples, config.dimension, config.n_clusters,
                             structural=config.structural, noise=config.noise,
                             ambiguity=config.ambiguity, random_seed=config.random_seed,
                             pipeline_steps=config.pipeline_steps)
    prof = profile_from_state(state, seed=config.random_seed or 0)
    X = state.Z.astype(np.float64)   # float64 so the profile is bit-reproducible from the file
    y = state.y.astype(np.int32)
    npz_path = out_dir / f"{name}.npz"
    np.savez_compressed(npz_path, X=X, y=y, difficulty_param=np.float32(prof.score),
                        seed=np.int32(config.random_seed if config.random_seed is not None else -1),
                        core=state.masks['core'], boundary=state.masks['boundary'],
                        internal_noise=state.masks['internal_noise'],
                        distant_noise=state.masks['distant_noise'],
                        confidence=state.confidence.astype(np.float32),
                        entropy=state.entropy.astype(np.float32))
    save_effective_config(config, out_dir / f"{name}.effective.json")
    with open(out_dir / f"{name}.profile.json", "w") as f:
        json.dump(prof.to_dict(), f, indent=2, default=float)
    s = state.summary()
    print(f"[saved] {npz_path.name}  n={X.shape[0]} d={X.shape[1]} K={len(np.unique(y[y >= 0]))}  "
          f"D={prof.score:.3f}  core={s['n_core']} boundary={s['n_boundary']} "
          f"noise={s['n_noise']} (internal={s['n_internal_noise']}, distant={s['n_distant_noise']})")
    return prof.to_dict()


# ---------------------------------------------------------------------------
def _load_labelled(args):
    if args.sklearn:
        from sklearn import datasets as skd
        from sklearn.preprocessing import StandardScaler
        d = getattr(skd, f"load_{args.sklearn}")()
        return StandardScaler().fit_transform(d.data), d.target, args.sklearn
    path = Path(args.path)
    if path.suffix == ".npz":
        z = np.load(path)
        return z["X"], z["y"], path.name
    import pandas as pd
    df = pd.read_csv(path)
    col = args.label_col or df.columns[-1]
    y = df[col].to_numpy()
    if y.dtype.kind not in "iu":
        _, y = np.unique(y, return_inverse=True)
    X = df.drop(columns=[col]).to_numpy(dtype=float)
    if args.standardize:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    return X, y.astype(int), path.name


def cmd_profile(args):
    X, y, name = _load_labelled(args)
    prof = difficulty_profile(X, y, relabel=args.relabel, k=args.k, seed=args.seed)
    d = prof.to_dict()
    print(f"# {name}: n={d['n_samples']} d={d['dimension']} K={d['n_clusters']} "
          f"noise={d['noise_fraction']:.2f}")
    print(f"D_dataset (v1 scalar) : {prof.score:.3f}")
    print("components            : " + "  ".join(f"{k}={v:.3f}" for k, v in prof.components.items()))
    print("knn profile           : " + "  ".join(f"{k}={v:.3f}" for k, v in prof.knn.items()))
    print("baselines             : " + "  ".join(f"{k}={v:.3f}" for k, v in prof.baselines.items()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(d, f, indent=2, default=float)
        print(f"[saved] {args.json}")


def cmd_generate(args):
    out_dir = Path(args.out)
    if args.config:
        raw = load_raw(args.config)
        for key, val in (("n_samples", args.n), ("dimension", args.d),
                         ("n_clusters", args.k), ("random_seed", args.seed)):
            raw.setdefault(key, val)
        config = config_from_dict(raw)
        name = args.name or f"{Path(args.config).stem}_s{config.random_seed}"
    elif args.preset:
        config = preset_config(args.preset, args.n, args.d, args.k, args.seed)
        name = args.name or f"{args.preset}_n{args.n}_d{args.d}_k{args.k}_s{args.seed}"
    else:
        config = DataGenerationConfig(n_samples=args.n, dimension=args.d, n_clusters=args.k,
                                      random_seed=args.seed)
        name = args.name or f"custom_n{args.n}_d{args.d}_k{args.k}_s{args.seed}"
    generate_and_save(config, name, out_dir)


def cmd_suite(args):
    from .io.suite import generate_all
    generate_all(out_dir=args.out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="clustaudit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate one dataset")
    g.add_argument("--config"); g.add_argument("--preset", choices=["easy", "medium", "hard"])
    g.add_argument("--n", type=int, default=1000); g.add_argument("--d", type=int, default=20)
    g.add_argument("--k", type=int, default=5); g.add_argument("--seed", type=int, default=42)
    g.add_argument("--name"); g.add_argument("--out", default="datasets")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("suite", help="generate the 75-dataset paper suite")
    s.add_argument("--out", default="datasets/suite")
    s.set_defaults(func=cmd_suite)

    p = sub.add_parser("profile", help="difficulty profile of a labelled dataset")
    p.add_argument("path", nargs="?", help=".npz (X, y) or .csv")
    p.add_argument("--sklearn", help="iris | wine | breast_cancer | digits ...")
    p.add_argument("--label-col"); p.add_argument("--standardize", action="store_true")
    p.add_argument("--relabel", action="store_true", help="smooth labels first (v1 real-data behaviour)")
    p.add_argument("--k", type=int, default=10); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="write the profile to this JSON file")
    p.set_defaults(func=cmd_profile)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
