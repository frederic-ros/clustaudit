"""
Evaluation of a partition against the STATE reference
=====================================================

One function, ``evaluate_partition``, returns every quantity the audit reads,
so that all benchmarks share one convention.

Conventions (these are load-bearing; v1 silently used only the third one):

ARI / NMI              on the ground-truth *inliers* (y_true >= 0). Points the
                       algorithm rejected as noise keep label -1, i.e. rejecting
                       a true inlier counts as an error. This is the primary
                       cluster-recovery measure.
ARI_full / NMI_full    on all points, noise label -1 treated as a class on both
                       sides (rewards correct noise detection).
ARI_kept / NMI_kept    v1 convention: restricted to the points the *algorithm*
                       kept (labels != -1). Favourable to density-based methods,
                       since they are scored only on what they chose to keep.
                       Reported for comparability with v1 only.
Silhouette / CH / DB   on the algorithm's non-noise points (standard practice).
noise_precision/recall of the algorithm's noise flag w.r.t. true noise
                       (NaN when the reference has no noise).
n_clusters_found       number of non-noise clusters returned.
frac_noise_pred        fraction of points the algorithm labelled -1.
degenerate             True when < 2 clusters or > 50% points rejected.
min_cluster_recall     recall of the *smallest* reference cluster: the largest fraction of
                       its points that one returned cluster captures. The quantity the
                       pair-counting ARI hides under size imbalance.
min_cluster_f1         F1 of the best-matching returned cluster for the smallest reference
                       cluster (recall is blind to merging; F1 is not). Precision counts every
                       point of the returned cluster, noise included, so the F1 also falls when
                       noise is attached to a rare cluster.
min_cluster_f1_inlier  the same with precision computed on ground-truth inliers only (v2.1.7),
                       i.e. the convention of the inlier ARI: falls only under absorption.
mean_cluster_f1        the same, averaged over reference clusters.
failure_mode           "ok" (ARI >= silent_threshold), "declared" (degenerate: the method
                       signals that it found no partition, by returning < 2 clusters or
                       rejecting most points) or "silent" (a well-formed partition is
                       returned and it is wrong). Default threshold 0.3.
"""

from __future__ import annotations
import numpy as np
from typing import Dict
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score)


def _ext(y, l):
    if len(y) < 2 or len(np.unique(l)) < 1:
        return np.nan, np.nan
    return adjusted_rand_score(y, l), normalized_mutual_info_score(y, l)


def _cluster_recalls(y_true, labels):
    """Per reference cluster: recall and F1 of the best-matching returned cluster.
    Recall alone is blind to merging (a rare cluster absorbed by a big one has recall 1);
    F1 = 2PR/(P+R) with P the share of the returned cluster that belongs to k."""
    rec, f1 = {}, {}
    for k in np.unique(y_true[y_true >= 0]):
        m = y_true == k
        preds = labels[m]
        preds = preds[preds != -1]
        if len(preds) == 0:
            rec[k] = f1[k] = 0.0
            continue
        best = np.bincount(preds).argmax()
        inter = np.sum(preds == best)
        r = inter / m.sum()
        p = inter / np.sum(labels == best)
        rec[k] = float(r); f1[k] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return rec, f1


def evaluate_partition(X: np.ndarray, y_true: np.ndarray, labels: np.ndarray,
                       silent_threshold: float = 0.3) -> Dict[str, float]:
    X = np.asarray(X); y_true = np.asarray(y_true); labels = np.asarray(labels)
    out: Dict[str, float] = {}
    kept = labels != -1
    inl = y_true >= 0
    n_found = len(np.unique(labels[kept])) if kept.any() else 0

    out["n_clusters_found"] = int(n_found)
    out["frac_noise_pred"] = float(np.mean(~kept))
    out["degenerate"] = bool(n_found < 2 or out["frac_noise_pred"] > 0.5)

    out["ARI"], out["NMI"] = _ext(y_true[inl], labels[inl])
    out["ARI_full"], out["NMI_full"] = _ext(y_true, labels)
    if kept.sum() > 1:
        out["ARI_kept"], out["NMI_kept"] = _ext(y_true[kept], labels[kept])
    else:
        out["ARI_kept"], out["NMI_kept"] = np.nan, np.nan

    if kept.sum() > 1 and n_found >= 2:
        out["Silhouette"] = float(silhouette_score(X[kept], labels[kept]))
        out["CH"] = float(calinski_harabasz_score(X[kept], labels[kept]))
        out["DB"] = float(davies_bouldin_score(X[kept], labels[kept]))
    else:
        out["Silhouette"] = out["CH"] = out["DB"] = np.nan

    rec, f1 = _cluster_recalls(y_true, labels)
    if rec:
        sizes = {k: int(np.sum(y_true == k)) for k in rec}
        smallest = min(sizes, key=sizes.get)
        out["min_cluster_recall"] = rec[smallest]
        out["min_cluster_f1"] = f1[smallest]
        out["mean_cluster_f1"] = float(np.mean(list(f1.values())))
        _, f1_inl = _cluster_recalls(y_true[inl], labels[inl])
        out["min_cluster_f1_inlier"] = f1_inl.get(smallest, np.nan)
    else:
        out["min_cluster_recall"] = out["min_cluster_f1"] = out["mean_cluster_f1"] = out["min_cluster_f1_inlier"] = np.nan
    ari = out["ARI"]
    out["failure_mode"] = ("declared" if out["degenerate"]
                           else ("silent" if (not np.isnan(ari) and ari < silent_threshold) else "ok"))

    true_noise = ~inl
    pred_noise = ~kept
    if true_noise.any():
        tp = np.sum(true_noise & pred_noise)
        out["noise_precision"] = float(tp / pred_noise.sum()) if pred_noise.any() else np.nan
        out["noise_recall"] = float(tp / true_noise.sum())
    else:
        out["noise_precision"] = out["noise_recall"] = np.nan
    return out


# column names, in the order benchmarks write them
RESULT_COLUMNS = ["ARI", "NMI", "ARI_full", "NMI_full", "ARI_kept", "NMI_kept",
                  "Silhouette", "CH", "DB", "n_clusters_found", "frac_noise_pred",
                  "degenerate", "noise_precision", "noise_recall",
                  "min_cluster_recall", "min_cluster_f1", "min_cluster_f1_inlier", "mean_cluster_f1", "failure_mode"]


def result_row(metrics: Dict[str, float]) -> Dict[str, float]:
    """Flatten ``evaluate_partition`` output with the legacy column names kept."""
    row = {k: metrics.get(k, np.nan) for k in RESULT_COLUMNS}
    row["DB_Index"] = row["DB"]; row["CH_Index"] = row["CH"]   # legacy names
    return row


def error_row(message: str) -> Dict[str, float]:
    """Result row of a run that raised: every measure is NaN, ``failure_mode`` is
    ``"error"``. Such a run is reported, never dropped; it is neither a success nor a
    failure of the method, and aggregates must state how they treat it."""
    row = {k: np.nan for k in RESULT_COLUMNS}
    row["DB_Index"] = row["CH_Index"] = np.nan
    row.update(failure_mode="error", run_status="error", run_error=str(message)[:300])
    return row


def evaluate_run(X: np.ndarray, y_true: np.ndarray, labels, status: str = "ok") -> Dict[str, float]:
    """One result row for one run, whatever happened to it.

    ``labels`` None (the method raised) gives :func:`error_row`; otherwise the partition
    is evaluated and ``run_status`` records how it was obtained (``"ok"`` or
    ``"no_admissible_configuration"``, the latter being a degenerate output)."""
    if labels is None or str(status).startswith("error"):
        return error_row(str(status).replace("error: ", "", 1))
    row = result_row(evaluate_partition(X, y_true, np.asarray(labels)))
    row.update(run_status=status, run_error="")
    return row
