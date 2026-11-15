"""
Compatibility shim: the tuners now live in ``clustaudit.audit.algorithms``.
This file keeps the v1 call signatures used by the experiment scripts
(``tune_agglomerative(X, "ward")``, ``tune_dbscan(X)`` ...) and exposes the
tuning protocol through the module-level CRITERION ("silhouette" | "dbcv" | "fixed").
"""
import numpy as np
from clustaudit.audit import algorithms as _a
from clustaudit.audit.evaluation import evaluate_partition, result_row

CRITERION = "silhouette"   # set from the scripts' --criterion flag

estimate_eps = _a.estimate_eps
get_labels = _a.get_labels


tuning_status = _a.tuning_status
from clustaudit.audit.evaluation import evaluate_run, error_row   # noqa: E402  (re-exported for the scripts)


class _TuningError:
    """Stands for a tuner that raised: running it raises the same error, so that the
    experiment loop records an error row for this method instead of losing the dataset."""
    tuning_status = "error"

    def __init__(self, exc):
        self.exc = exc

    def fit_predict(self, X):
        raise RuntimeError(f"tuning failed: {type(self.exc).__name__}: {self.exc}")


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return _TuningError(e)


def tune_kmeans(X, seed, n_clusters=None):        return _safe(_a.tune_kmeans, X, seed, n_clusters, CRITERION)
def tune_gmm(X, seed, n_clusters=None):           return _safe(_a.tune_gmm, X, seed, n_clusters, CRITERION)
def tune_spectral(X, seed, n_clusters=None):      return _safe(_a.tune_spectral, X, seed, n_clusters, CRITERION)
def tune_dbscan(X, n_clusters=None):              return _safe(_a.tune_dbscan, X, None, n_clusters, CRITERION)
def tune_hdbscan(X, n_clusters=None):             return _safe(_a.tune_hdbscan, X, None, n_clusters, CRITERION)
def tune_optics(X, n_clusters=None):              return _safe(_a.tune_optics, X, None, n_clusters, CRITERION)
def tune_dpc(X, n_clusters=None):                 return _safe(_a.tune_dpc, X, None, n_clusters, CRITERION)
def tune_agglomerative(X, linkage, n_clusters=None):
    return _safe(_a.tune_agglomerative, X, None, n_clusters, CRITERION, linkage)


def compute_metrics(X, y_true, labels):
    """v1 name; now returns the full evaluation (ARI = inlier ARI, see clustaudit.audit.evaluation)."""
    return evaluate_partition(X, y_true, labels)
