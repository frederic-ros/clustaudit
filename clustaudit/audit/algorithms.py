"""
Algorithm pool and tuning protocols
===================================

Every tuner has the same signature::

    tune_xxx(X, seed, n_clusters=None, criterion="silhouette") -> model with fit_predict

A tuner never returns ``None``. When no configuration of its grid yields an
admissible partition (at least two clusters among the points the model keeps, and a
computable criterion), it returns :class:`NoAdmissibleConfiguration`, whose output is
no partition (every point unassigned); the evaluation records it as a degenerate
output, i.e. a detectable failure. ``tuning_status(model)`` tells the two cases apart.

``criterion`` selects the *tuning protocol* (it is meant to be
explicit and comparable across families):

* ``"silhouette"`` : per-dataset grid search maximising the Silhouette on the
                     points the model keeps (v1 protocol, applied to every method).
* ``"dbcv"``       : same grid, but density-based methods (DBSCAN, HDBSCAN,
                     OPTICS) are tuned with DBCV (Moulavi et al. 2014) via
                     ``hdbscan.validity_index``; raises ImportError if the
                     package is missing (no silent fallback to the Silhouette).
* ``"fixed"``      : no per-dataset tuning. K-type methods receive the true K
                     (``n_clusters``), density methods use library defaults
                     with eps from the k-distance heuristic.

``ALGORITHMS`` maps a name to a tuner; ``run_method`` tunes and runs one of them and
returns ``(labels, status)``; ``run_pool`` runs a list of them on one dataset. No run
is ever dropped: a method that raises is reported with status ``"error"``.
"""

from __future__ import annotations
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, Optional

import numpy as np
from sklearn.cluster import (KMeans, DBSCAN, OPTICS, AgglomerativeClustering,
                             SpectralClustering, HDBSCAN)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

try:  # optional, for the DBCV protocol
    from hdbscan.validity import validity_index as _dbcv
except Exception:  # pragma: no cover
    _dbcv = None

DENSITY_BASED = {"DBSCAN", "HDBSCAN", "OPTICS", "DPC"}


# ---------------------------------------------------------------------------
def estimate_eps(X: np.ndarray, min_samples: int = 5) -> float:
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
    d, _ = nbrs.kneighbors(X)
    return float(np.percentile(d[:, -1], 90))


def get_labels(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "fit_predict"):
        return model.fit_predict(X)
    model.fit(X)
    return model.predict(X)


def _score(X, labels, criterion: str, density: bool) -> float:
    kept = labels != -1
    if kept.sum() < 2 or len(np.unique(labels[kept])) < 2:
        return -np.inf
    if criterion == "dbcv" and density:
        if _dbcv is None:
            raise ImportError("criterion='dbcv' requires the `hdbscan` package (pip install hdbscan); "
                              "refusing to fall back silently to the Silhouette")
        else:
            try:
                return float(_dbcv(X.astype(np.float64), labels))
            except Exception:
                return -np.inf
    return float(silhouette_score(X[kept], labels[kept]))


def _k_range(X):
    return range(2, min(10, X.shape[0] // 10))


class NoAdmissibleConfiguration:
    """Returned by a tuner when no configuration of its grid yields an admissible
    partition (at least two clusters among the retained points, and a computable
    criterion). Running it gives no partition: every point is left unassigned (-1).
    The evaluation records that output as degenerate, i.e. a detectable failure, so the
    run is kept in the results instead of disappearing from them."""
    tuning_status = "no_admissible_configuration"

    def __init__(self, n_tried: int = 0):
        self.n_tried = n_tried

    def fit_predict(self, X):
        return np.full(len(X), -1, dtype=int)

    def __repr__(self):
        return f"NoAdmissibleConfiguration(n_tried={self.n_tried})"


def tuning_status(model) -> str:
    """"ok", or "no_admissible_configuration" for the output of a failed tuning."""
    return getattr(model, "tuning_status", "ok")


def _best(models: Iterable, X, criterion, density=False):
    best_s, best_m, n = -np.inf, None, 0
    for m in models:
        n += 1
        try:
            s = _score(X, get_labels(m, X), criterion, density)
        except Exception:
            continue                      # a configuration that cannot run is not admissible
        if s > best_s:
            best_s, best_m = s, m
    return best_m if best_m is not None else NoAdmissibleConfiguration(n_tried=n)


# ---------------------------------------------------------------------------
def tune_kmeans(X, seed, n_clusters=None, criterion="silhouette"):
    if n_clusters is not None or criterion == "fixed":
        return KMeans(n_clusters=n_clusters or 5, random_state=seed, n_init=10)
    return _best((KMeans(n_clusters=k, random_state=seed, n_init=10) for k in _k_range(X)), X, criterion)


def tune_gmm(X, seed, n_clusters=None, criterion="silhouette"):
    if n_clusters is not None or criterion == "fixed":
        return GaussianMixture(n_components=n_clusters or 5, random_state=seed)
    return _best((GaussianMixture(n_components=k, random_state=seed) for k in _k_range(X)), X, criterion)


def tune_spectral(X, seed, n_clusters=None, criterion="silhouette"):
    mk = lambda k: SpectralClustering(n_clusters=k, random_state=seed, assign_labels="kmeans",
                                      affinity="nearest_neighbors")
    if n_clusters is not None or criterion == "fixed":
        return mk(n_clusters or 5)
    return _best((mk(k) for k in _k_range(X)), X, criterion)


def tune_agglomerative(X, seed=None, n_clusters=None, criterion="silhouette", linkage="ward"):
    if n_clusters is not None or criterion == "fixed":
        return AgglomerativeClustering(n_clusters=n_clusters or 5, linkage=linkage)
    return _best((AgglomerativeClustering(n_clusters=k, linkage=linkage) for k in _k_range(X)), X, criterion)


def tune_dbscan(X, seed=None, n_clusters=None, criterion="silhouette"):
    eps0 = estimate_eps(X)
    d = X.shape[1]
    if criterion == "fixed":
        return DBSCAN(eps=eps0, min_samples=max(5, d))
    grid = (DBSCAN(eps=eps0 * f, min_samples=max(2, int(m * d)))
            for f, m in product([0.5, 0.8, 1.0, 1.2, 1.5], [0.5, 1.0, 2.0, 3.0]))
    return _best(grid, X, criterion, density=True)


def tune_hdbscan(X, seed=None, n_clusters=None, criterion="silhouette"):
    if criterion == "fixed":
        return HDBSCAN(min_cluster_size=10, min_samples=5)
    return _best((HDBSCAN(min_cluster_size=s, min_samples=5) for s in [5, 10, 15, 20]),
                 X, criterion, density=True)


def tune_optics(X, seed=None, n_clusters=None, criterion="silhouette"):
    if criterion == "fixed":
        return OPTICS(min_samples=5, xi=0.05)
    return _best((OPTICS(min_samples=5, xi=xi) for xi in [0.03, 0.05, 0.08, 0.1]),
                 X, criterion, density=True)




# ---------------------------------------------------------------------------
# Density-peak clustering (Rodriguez & Laio, Science 2014), in-house implementation
# ---------------------------------------------------------------------------
from scipy.spatial.distance import pdist, squareform


class DensityPeaks:
    """Density peaks: rho_i = # points within d_c, delta_i = distance to the nearest
    denser point; centres = the K largest gamma = rho * delta; the other points
    follow their nearest denser neighbour. d_c = `percent` quantile of pairwise
    distances (Rodriguez & Laio recommend 1-2 %). Points whose density is below
    `noise_quantile` of rho AND farther than d_c from any denser point are labelled -1
    when `reject_noise` is True (a 'halo'-like rejection); default off."""

    def __init__(self, n_clusters=5, percent=2.0, reject_noise=False):
        self.n_clusters, self.percent, self.reject_noise = n_clusters, percent, reject_noise

    def fit_predict(self, X):
        X = np.asarray(X, float); n = len(X)
        D = squareform(pdist(X))
        dc = np.percentile(D[np.triu_indices(n, 1)], self.percent)
        rho = np.exp(-(D / dc) ** 2).sum(1) - 1.0            # gaussian kernel density
        order = np.argsort(-rho)
        delta = np.zeros(n); nn_up = np.full(n, -1)
        delta[order[0]] = D[order[0]].max()
        for i in range(1, n):
            p = order[i]; higher = order[:i]
            j = higher[np.argmin(D[p, higher])]
            delta[p] = D[p, j]; nn_up[p] = j
        gamma = rho * delta
        centres = np.argsort(-gamma)[: self.n_clusters]
        labels = np.full(n, -1)
        labels[centres] = np.arange(len(centres))
        for p in order:
            if labels[p] == -1 and nn_up[p] >= 0:
                labels[p] = labels[nn_up[p]]
        if self.reject_noise:
            halo = (rho < np.quantile(rho, 0.05)) & (delta > dc)
            labels[halo] = -1
        self.rho_, self.delta_, self.labels_ = rho, delta, labels
        return labels


def tune_dpc(X, seed=None, n_clusters=None, criterion="silhouette"):
    if n_clusters is not None or criterion == "fixed":
        return DensityPeaks(n_clusters=n_clusters or 5)
    return _best((DensityPeaks(n_clusters=k) for k in _k_range(X)), X, criterion, density=True)


ALGORITHMS: Dict[str, Callable] = {
    "KMeans":   tune_kmeans,
    "GMM":      tune_gmm,
    "Spectral": tune_spectral,
    "Ward":     lambda X, s, n=None, c="silhouette": tune_agglomerative(X, s, n, c, "ward"),
    "Average":  lambda X, s, n=None, c="silhouette": tune_agglomerative(X, s, n, c, "average"),
    "Complete": lambda X, s, n=None, c="silhouette": tune_agglomerative(X, s, n, c, "complete"),
    "DBSCAN":   tune_dbscan,
    "HDBSCAN":  tune_hdbscan,
    "OPTICS":   tune_optics,
    "DPC":      tune_dpc,
}
CLASSICAL_EIGHT = ["KMeans", "GMM", "Spectral", "Ward", "Average", "DBSCAN", "HDBSCAN", "OPTICS"]


def run_method(name: str, X: np.ndarray, seed: int, criterion: str = "silhouette",
               n_clusters: Optional[int] = None):
    """Tune and run one method of the pool. Returns ``(labels, status)``, status being
    ``"ok"`` or ``"no_admissible_configuration"``. Exceptions propagate: the caller
    records them (see :func:`run_pool` and ``clustaudit.audit.runner.audit_dataset``)."""
    model = ALGORITHMS[name](X, seed, n_clusters if criterion == "fixed" else None, criterion)
    return np.asarray(get_labels(model, X)), tuning_status(model)


def run_pool(X: np.ndarray, seed: int, names: Optional[Iterable[str]] = None,
             criterion: str = "silhouette", n_clusters: Optional[int] = None,
             with_status: bool = False) -> Dict[str, np.ndarray]:
    """Tune and run each named method on one dataset.

    Returns ``{name: labels}``; with ``with_status=True``, ``{name: (labels, status)}``.
    No method is dropped: one whose tuning finds no admissible configuration returns no
    partition (all -1, status ``"no_admissible_configuration"``), and one that raises
    has labels ``None`` and status ``"error: <message>"``. ``evaluate_run`` (in
    ``clustaudit.audit.evaluation``) turns either into a result row."""
    out = {}
    for name in (names or CLASSICAL_EIGHT):
        try:
            labels, status = run_method(name, X, seed, criterion, n_clusters)
        except Exception as e:
            warnings.warn(f"{name} failed: {e}")
            labels, status = None, f"error: {type(e).__name__}: {e}"
        out[name] = (labels, status) if with_status else labels
    return out
