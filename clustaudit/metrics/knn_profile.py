"""
k-NN structural difficulty profile
==================================

Local, label-aware descriptors of a clustering task computed on a k-nearest-
neighbour graph of ``(X, y)``. They complement (and are meant to be compared
against) the four v1 components ``S_sep, S_bnd, S_shape, S_noise``.

Design goals:

* **no sphericity prior** -- nothing here assumes convex or Gaussian clusters
  (the Silhouette does);
* **scale free** -- every quantity is a ratio or a fraction;
* **rank-robust** -- no covariance eigenvalue ratio;
* **cheap** -- one k-NN query, O(n log n) in practice, usable in high d;
* **interpretable** -- each descriptor answers one question.

The mechanisms (neighbourhood purity, mutual neighbours, fragmentation,
local density contrast) are those of the PDBI index
(Ros, Riad & Guillaume, Neurocomputing 2023), transposed from *partition
validation* to *task difficulty*; they are not a reimplementation of PDBI.

Descriptors (all in [0, 1], higher = harder, unless stated)
-----------------------------------------------------------
edge_impurity        fraction of k-NN edges (among labelled points) that join
                     two different classes. 0 = classes are k-NN separable.
ambiguous_fraction   fraction of labelled points whose neighbourhood purity
                     is < 0.5 (they are "surrounded" by another class).
mean_impurity        1 - mean neighbourhood purity.
fragmentation        mean over classes of 1 - |largest connected component| / n_k
                     on the *mutual* k-NN subgraph of the class. 0 = each class
                     is one connected piece; high = class split into islands
                     (what Average linkage and density methods trip on).
density_heterogeneity  coefficient of variation of the per-class local
                     density (1 / mean k-NN radius), clipped to [0, 1].
                     Captures inter-cluster density contrast.
intra_density_spread  mean over classes of the within-class CV of local
                     density, clipped. Captures core/halo structure.
noise_fraction       fraction of points with label -1.
noise_proximity      for noise points: median of r_k(noise) / median r_k(labelled),
                     mapped through 1 / (1 + x): ~1 when noise sits inside the
                     clusters' support (ambiguous), ~0 when it is far away.
                     NaN when there is no noise.
intrinsic_dim_twonn  TwoNN estimate of the intrinsic dimension (Facco et al.
                     2017), *not* normalised; reported for information.

``knn_profile`` returns a plain dict so it can be dropped into a DataFrame.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def _twonn_id(dists: np.ndarray) -> float:
    """TwoNN intrinsic dimension from sorted NN distances (cols 1 and 2 used)."""
    r1, r2 = dists[:, 1], dists[:, 2]
    ok = (r1 > 0) & (r2 > r1)
    if ok.sum() < 10:
        return np.nan
    mu = r2[ok] / r1[ok]
    # discard the top 10% (Facco et al.) for robustness
    mu = np.sort(mu)[: int(0.9 * len(mu))]
    return float(len(mu) / np.sum(np.log(mu)))


def knn_profile(X: np.ndarray, y: np.ndarray, k: int = 10,
                metric: str = "euclidean") -> Dict[str, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(y)
    labelled = y >= 0
    Xl, yl = X[labelled], y[labelled]
    nl = len(yl)
    classes = np.unique(yl)
    out: Dict[str, float] = {}
    if nl < k + 2 or len(classes) < 2:
        keys = ["edge_impurity", "ambiguous_fraction", "mean_impurity", "fragmentation",
                "density_heterogeneity", "intra_density_spread", "noise_fraction",
                "noise_proximity", "intrinsic_dim_twonn"]
        return {kk: np.nan for kk in keys}

    k_eff = min(k, nl - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric).fit(Xl)
    dist, ind = nn.kneighbors(Xl)            # column 0 is self
    nbr = ind[:, 1:]
    r_k = dist[:, -1]                        # k-NN radius

    # --- purity -------------------------------------------------------
    same = (yl[nbr] == yl[:, None])
    purity = same.mean(axis=1)
    out["edge_impurity"] = float(1.0 - same.mean())
    out["ambiguous_fraction"] = float(np.mean(purity < 0.5))
    out["mean_impurity"] = float(1.0 - purity.mean())

    # --- fragmentation on the mutual k-NN graph -------------------------
    rows = np.repeat(np.arange(nl), k_eff)
    cols = nbr.ravel()
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nl, nl)).tocsr()
    M = A.multiply(A.T)                      # mutual edges
    frag = []
    for c in classes:
        idx = np.where(yl == c)[0]
        if len(idx) < 2:
            frag.append(0.0)
            continue
        sub = M[idx][:, idx]
        ncomp, lab = connected_components(sub, directed=False)
        largest = np.bincount(lab).max()
        frag.append(1.0 - largest / len(idx))
    out["fragmentation"] = float(np.mean(frag))

    # --- local density ---------------------------------------------------
    rho = 1.0 / (r_k + 1e-12)
    class_rho = np.array([rho[yl == c].mean() for c in classes])
    out["density_heterogeneity"] = float(np.clip(class_rho.std() / (class_rho.mean() + 1e-12), 0, 1))
    intra = [rho[yl == c].std() / (rho[yl == c].mean() + 1e-12) for c in classes if (yl == c).sum() > 1]
    out["intra_density_spread"] = float(np.clip(np.mean(intra), 0, 1)) if intra else 0.0

    # --- noise -------------------------------------------------------------
    out["noise_fraction"] = float(np.mean(~labelled))
    if np.any(~labelled):
        d_noise = nn.kneighbors(X[~labelled], n_neighbors=k_eff)[0][:, -1]
        ratio = np.median(d_noise) / (np.median(r_k) + 1e-12)
        out["noise_proximity"] = float(1.0 / (1.0 + ratio))
    else:
        out["noise_proximity"] = 0.0   # no noise: nothing is close to any inlier (v2.1.7; was NaN)

    # --- intrinsic dimension ----------------------------------------------
    out["intrinsic_dim_twonn"] = _twonn_id(dist) if k_eff >= 2 else np.nan
    return out


def knn_profile_vector(profile: Dict[str, float]) -> np.ndarray:
    """The normalised descriptors as a vector (drops the un-normalised ID)."""
    keys = ["edge_impurity", "ambiguous_fraction", "mean_impurity", "fragmentation",
            "density_heterogeneity", "intra_density_spread", "noise_fraction", "noise_proximity"]
    return np.array([profile[k] for k in keys], dtype=float)
