"""
Calibrated generation: ask for a *measured* difficulty, not a parameter.

The parametric generator maps parameters theta to a dataset G(theta, z); the profile H
measures the dataset afterwards. Calibration closes the loop: given a target h* on a few
measured descriptors, it searches theta (for a fixed seed z) so that H(G(theta, z)) is
within a tolerance of h*.

    theta* = argmin_theta || W (H(G(theta, z)) - h*) ||^2     s.t. the generator is valid

The search is deliberately simple (coordinate-wise random search with shrinking steps on
a handful of generator parameters), because the point is the principle, not the optimizer:
a dataset is accepted on the difficulty it *has*, not on the one its parameters suggest.

    from clustaudit.generators.calibrate import calibrate
    res = calibrate({"knn_edge_impurity": 0.15, "S_noise": 0.4}, seed=42)
    res.state, res.profile, res.params, res.error, res.n_evals

Searchable parameters (``SPACE``): the cluster spread sigma, the anisotropy alpha, the
noise amount eta and the noise placement eta_int. Everything else stays at the frozen
operating point of the sweeps, so that calibrated and parametric datasets differ only
in how their difficulty was chosen.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..config.params import StructuralParams, NoiseParams, AmbiguityParams, DensityParams
from ..core.pipeline import generate_dataset
from ..metrics.profile import profile_from_state

# name -> (lower, upper, setter)
SPACE = {
    "sigma":   (0.10, 1.00, lambda c, v: setattr(c["structural"], "cluster_std_range", (float(v), float(v)))),
    "alpha":   (1.00, 8.00, lambda c, v: setattr(c["structural"], "anisotropy_factor_range", (float(v), float(v)))),
    "eta":     (0.00, 0.40, lambda c, v: setattr(c["noise"], "noise_ratio", float(v))),
    "eta_int": (0.00, 1.00, lambda c, v: setattr(c["noise"], "internal_noise_ratio", float(v))),
}
DEFAULT_START = {"sigma": 0.2, "alpha": 2.0, "eta": 0.1, "eta_int": 0.5}


def operating_point() -> Dict:
    """The frozen medium operating point of the sweeps (same as audit.runner._base_config)."""
    return {
        "structural": StructuralParams(hypercube_connectivity=0.5, enable_anisotropy=True,
                                       anisotropy_factor_range=(2.0, 2.0), enable_rotation=True,
                                       cluster_std_range=(0.15, 0.25), cluster_size_imbalance=1.0,
                                       density_params=DensityParams(0.0, intra_density_strength=0.0)),
        "noise": NoiseParams(noise_ratio=0.10, internal_noise_ratio=0.5),
        "ambiguity": AmbiguityParams(),
    }


@dataclass
class CalibrationResult:
    params: Dict[str, float]
    profile: Dict[str, float]
    error: float                      # weighted RMS distance to the target, in descriptor units
    n_evals: int
    hit: bool                         # every target descriptor within its tolerance
    state: object = None
    history: list = field(default_factory=list)


def _measure(params: Dict[str, float], n: int, d: int, K: int, seed: int, cfg_base: Dict):
    cfg = copy.deepcopy(cfg_base)
    for name, v in params.items():
        SPACE[name][2](cfg, v)
    state = generate_dataset(n, d, K, structural=cfg["structural"], noise=cfg["noise"],
                             ambiguity=cfg["ambiguity"], random_seed=seed)
    prof = profile_from_state(state, seed=seed).to_dict()
    for k, v in prof.items():   # a degenerate measurement (e.g. no core left) must not be a target hit
        if isinstance(v, float) and np.isnan(v): prof[k] = np.inf
    return state, prof


def calibrate(target: Dict[str, float], seed: int = 0, n: int = 1000, d: int = 20, K: int = 5,
              tolerance: Optional[Dict[str, float]] = None, weights: Optional[Dict[str, float]] = None,
              budget: int = 40, start: Optional[Dict[str, float]] = None,
              search: Tuple[str, ...] = ("sigma", "alpha", "eta", "eta_int"),
              rng_seed: int = 0, keep_state: bool = True) -> CalibrationResult:
    """
    Search the generator parameters so that the measured profile reaches ``target``.

    target     : {descriptor_name: value}, names as in DifficultyProfile.to_dict()
                 (e.g. "knn_edge_impurity", "S_noise", "knn_noise_proximity", "S_separation").
    tolerance  : {descriptor_name: abs tolerance}; default 0.05 each. ``hit`` is True when all are met.
    weights    : relative weights in the objective; default 1 / tolerance.
    budget     : maximum number of generate-and-measure evaluations.
    search     : which parameters of SPACE may move; the others stay at the operating point.

    Strategy: evaluate the start point, then rounds of coordinate-wise proposals with a step that
    halves whenever a full round brings no improvement (a derivative-free pattern search), with one
    random restart draw per round so that a bad start does not trap the search.
    """
    tol = {k: (tolerance or {}).get(k, 0.05) for k in target}
    w = {k: (weights or {}).get(k, 1.0 / tol[k]) for k in target}
    rng = np.random.RandomState(rng_seed)
    cfg_base = operating_point()
    cur = dict(DEFAULT_START); cur.update(start or {}); cur = {k: cur[k] for k in search}
    lo = {k: SPACE[k][0] for k in search}; hi = {k: SPACE[k][1] for k in search}
    step = {k: 0.25 * (hi[k] - lo[k]) for k in search}

    def objective(prof):
        return float(np.sqrt(np.mean([(w[k] * (prof[k] - target[k])) ** 2 for k in target])))

    def is_hit(prof):
        return all(abs(prof[k] - target[k]) <= tol[k] for k in target)

    history = []
    state, prof = _measure(cur, n, d, K, seed, cfg_base); best = (objective(prof), dict(cur), prof, state); n_evals = 1
    history.append((dict(cur), best[0]))
    while n_evals < budget and not is_hit(best[2]):
        improved = False
        for k in search:
            for sgn in (+1, -1):
                if n_evals >= budget: break
                cand = dict(best[1]); cand[k] = float(np.clip(cand[k] + sgn * step[k], lo[k], hi[k]))
                if cand[k] == best[1][k]: continue
                st, pr = _measure(cand, n, d, K, seed, cfg_base); n_evals += 1; f = objective(pr); history.append((cand, f))
                if f < best[0]:
                    best = (f, cand, pr, st); improved = True; break
            if is_hit(best[2]): break
        if not improved and n_evals < budget:
            # halve the steps and try one random draw around the best point
            for k in search: step[k] *= 0.5
            cand = {k: float(np.clip(best[1][k] + rng.normal(0, 2 * step[k]), lo[k], hi[k])) for k in search}
            st, pr = _measure(cand, n, d, K, seed, cfg_base); n_evals += 1; f = objective(pr); history.append((cand, f))
            if f < best[0]: best = (f, cand, pr, st)
            if max(step.values()) < 1e-3: break
    f, params, prof, state = best
    return CalibrationResult(params=params, profile=prof, error=f, n_evals=n_evals, hit=is_hit(prof),
                             state=state if keep_state else None, history=history)
