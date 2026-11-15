"""
The audit interface
===================

A clustering method is anything that maps ``(X, seed) -> labels`` (``-1`` for
rejected points). ``audit`` runs it on a suite of difficulty-controlled
datasets and returns one row per dataset with the full difficulty profile and
the full evaluation, ready for correlation / threshold analysis::

    from clustaudit.audit import audit, sweep
    df = audit(lambda X, seed: KMeans(5, random_state=seed).fit_predict(X),
               suite_dir="datasets/suite")

``sweep`` builds a one-factor sweep on the fly (anisotropy, noise, internal
noise fraction, overlap (cluster std), connectivity, dimensionality, imbalance, densities) with every other
generator parameter held fixed, which is the interventional protocol of the paper::

    df = sweep(my_algo, factor="eta_int", values=[0, .25, .5, .75, 1], seeds=[42, 123])
"""

from __future__ import annotations
import copy
import glob
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..config.params import StructuralParams, NoiseParams, AmbiguityParams, DensityParams
from ..core.pipeline import generate_dataset
from ..metrics import profile_from_state, difficulty_profile
from .evaluation import evaluate_partition, result_row, evaluate_run, error_row

Algo = Callable[[np.ndarray, int], np.ndarray]


def _base_config() -> Dict:
    """The frozen 'medium' operating point used by every single-factor sweep."""
    return {
        "structural": StructuralParams(hypercube_connectivity=0.5, enable_anisotropy=True,
                                       anisotropy_factor_range=(2.0, 2.0), enable_rotation=True,
                                       cluster_std_range=(0.15, 0.25), cluster_size_imbalance=1.0,
                                       density_params=DensityParams(0.0, intra_density_strength=0.0)),
        "noise": NoiseParams(noise_ratio=0.10, internal_noise_ratio=0.5),
        "ambiguity": AmbiguityParams(),
    }


SWEEPS = {
    "anisotropy":  lambda c, v: setattr(c["structural"], "anisotropy_factor_range", (float(v), float(v))),
    "eta":         lambda c, v: setattr(c["noise"], "noise_ratio", float(v)),
    "eta_int":     lambda c, v: setattr(c["noise"], "internal_noise_ratio", float(v)),
    "overlap":     lambda c, v: setattr(c["structural"], "cluster_std_range", (float(v), float(v))),
    "connectivity": lambda c, v: setattr(c["structural"], "hypercube_connectivity", float(v)),
    "imbalance":   lambda c, v: setattr(c["structural"], "cluster_size_imbalance", float(v)),
    "inter_density": lambda c, v: setattr(c["structural"].density_params, "inter_density_variation", float(v)),
    "intra_density": lambda c, v: (setattr(c["structural"].density_params, "intra_density_mode", "core_dense"),
                                   setattr(c["structural"].density_params, "intra_density_strength", float(v))),
    "dimension":   None,   # handled through the `dimension` argument
}


def audit_dataset(algos: Dict[str, Algo], X: np.ndarray, y: np.ndarray, seed: int,
                  profile=None, extra: Optional[Dict] = None) -> List[Dict]:
    """Run every algorithm on one dataset; exactly one row per algorithm.

    ``fn(X, seed)`` returns labels, or ``(labels, status)`` (as ``run_method`` does).
    A method that raises is not dropped: its row has ``failure_mode = "error"``,
    ``run_status = "error"`` and the message in ``run_error`` (see ``error_row``)."""
    prof = profile or difficulty_profile(X, y, seed=seed)
    pd_ = prof.to_dict()
    rows = []
    for name, fn in algos.items():
        try:
            out = fn(X, seed)
            labels, status = out if isinstance(out, tuple) else (out, "ok")
            res = evaluate_run(X, y, labels, status)
        except Exception as e:
            print(f"  ! {name} failed: {type(e).__name__}: {e} (recorded as an error row)")
            res = error_row(f"{type(e).__name__}: {e}")
        rows.append({"algorithm": name, "seed": seed, **(extra or {}), **pd_, **res})
    return rows


def audit(algo: Algo | Dict[str, Algo], suite_dir: str = "datasets/suite",
          name: str = "method") -> pd.DataFrame:
    """Audit on a stored suite of .npz files (X, y, seed)."""
    algos = algo if isinstance(algo, dict) else {name: algo}
    rows = []
    for f in sorted(glob.glob(str(Path(suite_dir) / "*.npz"))):
        z = np.load(f)
        seed = int(z["seed"]) if "seed" in z else 0
        rows += audit_dataset(algos, z["X"].astype(float), z["y"].astype(int), seed,
                              extra={"dataset": Path(f).name})
    return pd.DataFrame(rows)


def sweep(algo: Algo | Dict[str, Algo], factor: str, values: Iterable, seeds: Iterable[int] = (42, 123, 456),
          n_samples: int = 1000, dimension: int = 20, n_clusters: int = 5, fixed_n: bool = True,
          base: Optional[Dict] = None, name: str = "method") -> pd.DataFrame:
    """
    Single-factor interventional sweep. ``factor`` in SWEEPS. For
    ``factor == "dimension"``, ``fixed_n=False`` scales n = max(n_samples, 50 d)
    (n grows with d); ``fixed_n=True`` keeps n constant (the control).
    """
    if factor not in SWEEPS:
        raise ValueError(f"factor must be one of {list(SWEEPS)}")
    algos = algo if isinstance(algo, dict) else {name: algo}
    rows = []
    for v in values:
        for seed in seeds:
            cfg = copy.deepcopy(base or _base_config())
            d, n = dimension, n_samples
            if factor == "dimension":
                d = int(v)
                n = n_samples if fixed_n else max(n_samples, 50 * d)
            else:
                SWEEPS[factor](cfg, v)
            state = generate_dataset(n, d, n_clusters, structural=cfg["structural"],
                                     noise=cfg["noise"], ambiguity=cfg["ambiguity"], random_seed=seed)
            try:
                prof = profile_from_state(state, seed=seed)
            except ValueError as e:
                print(f"  ! {factor}={v} seed={seed}: {e} (skipped)")
                continue
            rows += audit_dataset(algos, state.Z, state.y, seed, profile=prof,
                                  extra={"factor": factor, "value": v, "n_samples": n, "dimension_swept": d})
    return pd.DataFrame(rows)
