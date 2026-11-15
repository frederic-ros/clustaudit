"""
Difficulty profile of a labelled dataset
========================================

Single entry point used by *every* benchmark (synthetic, cross-generator,
real data, deep). There is no longer a stand-alone copy of the scoring chain:
``difficulty_profile`` is the chain.

    profile = difficulty_profile(X, y)        # any labelled data
    profile = profile_from_state(state)       # a generated STATE

A ``DifficultyProfile`` carries

* ``components`` : the four v1 components S_sep, S_bnd, S_shape, S_noise
  (computed through the STATE reconstruction: classifier -> confidence ->
  core / boundary / noise masks -> metrics);
* ``knn``        : the k-NN structural descriptors of ``knn_profile``;
* ``score``      : the v1 scalar D_dataset = fixed-weight sum of the four
  components. Kept for continuity and as *one* possible summary; the paper
  no longer presents it as a contribution;
* ``baselines``  : simple label-based separability baselines that one
  asked to be compared against (1 - Silhouette on the reference labels,
  cross-validated k-NN error, between/within distance ratio).

Everything is returned as plain floats so a profile is one row of a table.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from ..core.state import ClusteringState
from ..config.params import AmbiguityParams
from ..generators.refinement import refine_labels
from .metricsstate import compute_complete_metrics
from .complexity import compute_complexity_score
from .knn_profile import knn_profile


def score_knn(components: Dict[str, float], knn: Dict[str, float]) -> float:
    """Silhouette-free scalar: equal-weight mean of k-NN edge impurity (capped at 0.5),
    the boundary component and the noise component. No shape prior, no fitted constant."""
    imp = min(knn.get("edge_impurity", 0.0) / 0.5, 1.0) if not np.isnan(knn.get("edge_impurity", np.nan)) else 0.0
    return float((imp + components["boundary"] + components["noise"]) / 3.0)


@dataclass
class DifficultyProfile:
    score: float
    components: Dict[str, float]
    knn: Dict[str, float]
    baselines: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    n_clusters: int = 0
    dimension: int = 0
    noise_fraction: float = 0.0

    @property
    def score_knn(self) -> float:
        return score_knn(self.components, self.knn)

    def to_dict(self, prefix_components: str = "S_", prefix_knn: str = "knn_",
                prefix_baseline: str = "base_") -> Dict[str, float]:
        d = {"D_dataset": self.score, "D_knn": self.score_knn, "n_samples": self.n_samples,
             "n_clusters": self.n_clusters, "dimension": self.dimension,
             "noise_fraction": self.noise_fraction}
        d.update({prefix_components + k: v for k, v in self.components.items()})
        d.update({prefix_knn + k: v for k, v in self.knn.items()})
        d.update({prefix_baseline + k: v for k, v in self.baselines.items()})
        return d

    def __repr__(self) -> str:
        c = ", ".join(f"{k}={v:.2f}" for k, v in self.components.items())
        return f"DifficultyProfile(D={self.score:.3f}; {c}; knn_edge_impurity={self.knn.get('edge_impurity', float('nan')):.2f})"


# ---------------------------------------------------------------------------
def state_from_labeled_data(X: np.ndarray, y: np.ndarray, seed: int = 0) -> ClusteringState:
    """Wrap raw ``(X, y)`` (``-1`` = noise) into a STATE with empty masks."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    labelled = y >= 0
    return ClusteringState(
        Z=X, y=y,
        masks={'core': labelled.copy(), 'boundary': np.zeros(n, bool),
               'noise': ~labelled, 'internal_noise': np.zeros(n, bool),
               'distant_noise': np.zeros(n, bool)},
        entropy=np.zeros(n), confidence=np.ones(n),
        metadata={'source': 'external'}, timestamp=datetime.now(), random_seed=seed)


def _baselines(X: np.ndarray, y: np.ndarray, seed: int, k: int) -> Dict[str, float]:
    labelled = y >= 0
    Xl, yl = X[labelled], y[labelled]
    out: Dict[str, float] = {}
    classes, counts = np.unique(yl, return_counts=True)
    if len(classes) < 2 or len(yl) < 10:
        return {"one_minus_silhouette": np.nan, "knn_cv_error": np.nan, "within_between_ratio": np.nan}
    try:
        out["one_minus_silhouette"] = float(1.0 - silhouette_score(Xl, yl))
    except ValueError:
        out["one_minus_silhouette"] = np.nan
    folds = int(min(5, counts.min()))
    if folds >= 2:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        acc = cross_val_score(KNeighborsClassifier(n_neighbors=min(k, len(yl) - 1)), Xl, yl, cv=skf)
        out["knn_cv_error"] = float(1.0 - acc.mean())
    else:
        out["knn_cv_error"] = np.nan
    cents = np.array([Xl[yl == c].mean(axis=0) for c in classes])
    within = np.mean([np.linalg.norm(Xl[yl == c] - cents[i], axis=1).mean() for i, c in enumerate(classes)])
    dc = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)
    between = dc[np.triu_indices(len(classes), 1)].mean()
    out["within_between_ratio"] = float(within / (between + 1e-12))
    return out


def profile_from_state(state: ClusteringState, k: int = 10, seed: int = 0,
                       with_baselines: bool = True, use_state_masks: bool = False,
                       ambiguity: Optional[AmbiguityParams] = None) -> DifficultyProfile:
    """
    Profile of a generated STATE.

    By default the STATE is *re-measured* with the canonical ``AmbiguityParams``
    (same classifier, thresholds and folds for every dataset, whatever Theta_amb
    the generator used), so that profiles are commensurable across regimes and
    identical to ``difficulty_profile(state.Z, state.y)``. Set
    ``use_state_masks=True`` to read the masks the generator left in the STATE.
    """
    if not use_state_masks:
        return difficulty_profile(state.Z, state.y, ambiguity=ambiguity, relabel=False,
                                  k=k, seed=seed, with_baselines=with_baselines)
    metrics = compute_complete_metrics(state, include_noise=True)
    comp = compute_complexity_score(metrics)
    knn = knn_profile(state.Z, state.y, k=k)
    base = _baselines(state.Z, state.y, seed, k) if with_baselines else {}
    return DifficultyProfile(
        score=comp['complexity_score'], components=dict(comp['components']), knn=knn,
        baselines=base, n_samples=state.n_samples(), n_clusters=state.n_clusters(),
        dimension=state.dimension(), noise_fraction=float(np.mean(state.y < 0)))


def difficulty_profile(X: np.ndarray, y: np.ndarray, ambiguity: Optional[AmbiguityParams] = None,
                       relabel: bool = False, k: int = 10, seed: int = 0,
                       with_baselines: bool = True) -> DifficultyProfile:
    """
    Difficulty profile of any labelled dataset ``(X, y)``.

    The STATE is *reconstructed*: a classifier is fitted on the labelled points,
    out-of-sample confidence categorises them as core / boundary, noise points
    (``y == -1``) are split into internal / distant geometrically, and the
    components follow. This is exactly the chain applied to generated data, so
    scores are comparable across sources.

    relabel : if True, run the pre-noise refinement first, i.e. replace ``y`` by
              a self-consistent smoothed version (v1 behaviour on real data).
              Default False: external labels are taken as they are.
    """
    ambiguity = ambiguity or AmbiguityParams()
    rng = np.random.RandomState(seed)
    state = state_from_labeled_data(X, y, seed)
    if len(np.unique(state.y[state.y >= 0])) < 2:
        raise ValueError("difficulty_profile needs at least two labelled classes")
    if relabel:
        state = refine_labels(state, ambiguity.pre_refinement_classifier, ambiguity, "pre-noise", rng)
    state = refine_labels(state, ambiguity.post_refinement_classifier, ambiguity, "post-noise", rng)
    return profile_from_state(state, k=k, seed=seed, with_baselines=with_baselines, use_state_masks=True)


def compute_D_dataset(X: np.ndarray, y: np.ndarray, **kw) -> float:
    """Backward-compatible scalar accessor (used by the real-data benchmark)."""
    return difficulty_profile(X, y, **kw).score
