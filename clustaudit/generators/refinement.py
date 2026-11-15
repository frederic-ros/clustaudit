"""
Label Refiner: supervised label refinement and ambiguity quantification
=======================================================================

Two phases share this module.

* ``pre-noise``  : the freshly generated labels are made *self-consistent* by
  iterating a classifier until it reproduces them (accuracy >= threshold). The
  labels returned by this phase become the reference partition ``y``. This is
  a deliberate design choice (the reference must be geometrically well formed
  for label-based evaluation to be legitimate) and it is declared as such.

* ``post-noise`` : the classifier is fitted on the labelled points only and
  used to score every point. Confidence (max posterior) and entropy define the
  core / boundary categories; noise points (``y == -1``) are split into
  *internal* and *distant* by a geometric criterion.

Changes with respect to v1 (see CHANGELOG):
  - confidence of labelled points is out-of-sample by default
    (``AmbiguityParams.confidence_mode == "cv"``);
  - the internal/distant split of noise no longer relies on
    ``noise_threshold`` (unreachable with a K-class posterior) but on the
    k-NN distance to the labelled set (``distant_noise_quantile``).
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from typing import Literal
from ..core.state import ClusteringState
from ..config.params import AmbiguityParams


def _make_classifier(classifier_type: str, n_train: int, params: AmbiguityParams,
                     random_state: np.random.RandomState):
    if classifier_type == "knn":
        k = params.pre_refinement_k_neighbors or int(np.sqrt(n_train))
        k = max(1, min(k, n_train - 1))
        return KNeighborsClassifier(n_neighbors=k), k
    if classifier_type == "svm":
        return SVC(probability=True, random_state=random_state.randint(0, 2**31)), None
    return RandomForestClassifier(n_estimators=100,
                                  random_state=random_state.randint(0, 2**31)), None


def _entropy(proba: np.ndarray, params: AmbiguityParams) -> np.ndarray:
    eps = 1e-10
    if params.entropy_method == "shannon":
        e = -np.sum(proba * np.log(proba + eps), axis=1)
        if params.entropy_normalization and proba.shape[1] > 1:
            e = e / np.log(proba.shape[1])
        return e
    if params.entropy_method == "gini":
        return 1 - np.sum(proba ** 2, axis=1)
    e = -np.log(np.sum(proba ** 2, axis=1) + eps)  # renyi
    if params.entropy_normalization and proba.shape[1] > 1:
        e = e / -np.log(1 / proba.shape[1])
    return e


def refine_labels(
    state: ClusteringState,
    classifier_type: Literal["knn", "svm", "random_forest"],
    params: AmbiguityParams,
    phase: Literal["pre-noise", "post-noise"],
    random_state: np.random.RandomState
) -> ClusteringState:
    """Refine labels (pre-noise) or score ambiguity (post-noise). See module doc."""
    is_pre = (phase == "pre-noise")
    if (is_pre and not params.enable_pre_refinement) or \
       (not is_pre and not params.enable_post_refinement):
        return state
    if state.n_samples() == 0:
        return state

    labelled = state.y >= 0
    if not np.any(labelled):
        return state
    X_train, y_train = state.Z[labelled], state.y[labelled]
    if len(np.unique(y_train)) < 2:
        return state

    clf, k_nn = _make_classifier(classifier_type, len(y_train), params, random_state)

    # ------------------------------------------------------------------
    # Pre-noise: iterate until the classifier reproduces the labels
    # ------------------------------------------------------------------
    if is_pre:
        n_classes0 = len(np.unique(y_train))
        for _ in range(10):
            clf.fit(X_train, y_train)
            y_pred_train = clf.predict(X_train)
            if np.mean(y_pred_train == y_train) >= params.pre_refinement_threshold:
                break
            # never let the smoothing erase a cluster: under heavy overlap the
            # classifier can absorb a whole class; keep the last labelling instead
            if len(np.unique(y_pred_train)) < n_classes0:
                break
            y_train = y_pred_train
        clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Confidence / entropy for labelled points
    # ------------------------------------------------------------------
    n_classes_train = len(np.unique(y_train))
    use_cv = (params.confidence_mode == "cv" and not is_pre
              and n_classes_train >= 2
              and np.min(np.bincount(y_train)) >= params.cv_folds)
    if use_cv:
        # folds are seeded from the data themselves, so the same (X, y) gives the
        # same confidence whether it comes from the generator or from outside
        cv_seed = int(np.sum(np.abs(X_train)) * 1e6) % (2**31 - 1)
        skf = StratifiedKFold(n_splits=params.cv_folds, shuffle=True, random_state=cv_seed)
        proba_lab = cross_val_predict(clf, X_train, y_train, cv=skf, method="predict_proba")
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train)
        proba_lab = clf.predict_proba(X_train)

    proba = np.zeros((state.n_samples(), proba_lab.shape[1]))
    proba[labelled] = proba_lab
    if np.any(~labelled):
        proba[~labelled] = clf.predict_proba(state.Z[~labelled])

    y_pred = np.full(state.n_samples(), -1, dtype=int)
    classes = clf.classes_
    y_pred[:] = classes[np.argmax(proba, axis=1)]

    confidence = np.max(proba, axis=1)
    entropy = _entropy(proba, params)

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    new_masks = state.masks.copy()
    new_masks['core'] = (confidence >= params.core_threshold) & labelled
    new_masks['boundary'] = (confidence < params.core_threshold) & labelled

    if not is_pre:
        noise_mask = state.y == -1
        if np.any(noise_mask):
            # geometric split. Gap scale = low quantile of the distance from a
            # labelled point to the nearest point of *another* cluster (i.e. the
            # width of the inter-cluster gaps, measured on the points that face
            # them). A noise point closer to the data than that is *internal*
            # (it sits inside or between cluster supports, where assignment is
            # ambiguous); farther, it is a *distant* outlier.
            classes = np.unique(y_train)
            nn_other = np.full(len(y_train), np.inf)
            if len(classes) > 1:
                for c in classes:
                    other = y_train != c
                    nn_c = NearestNeighbors(n_neighbors=1).fit(X_train[other])
                    nn_other[~other] = nn_c.kneighbors(X_train[~other])[0][:, 0]
                gap = np.quantile(nn_other, params.distant_noise_quantile)
            else:
                nn_all = NearestNeighbors(n_neighbors=2).fit(X_train)
                gap = np.quantile(nn_all.kneighbors(X_train)[0][:, 1], 0.99)
            d1 = NearestNeighbors(n_neighbors=1).fit(X_train).kneighbors(state.Z[noise_mask])[0][:, 0]
            distant = np.zeros(state.n_samples(), dtype=bool)
            distant[np.where(noise_mask)[0][d1 > gap]] = True
            new_masks['distant_noise'] = distant
            new_masks['internal_noise'] = noise_mask & ~distant
        else:
            new_masks['distant_noise'] = np.zeros(state.n_samples(), dtype=bool)
            new_masks['internal_noise'] = np.zeros(state.n_samples(), dtype=bool)

    # pre-noise: the refined labels become the reference partition
    if is_pre:
        y_out = state.y.copy()
        y_out[labelled] = y_train
        state.metadata["n_relabelled"] = int(np.sum(y_train != state.y[labelled]))
        state.metadata["frac_relabelled"] = float(np.mean(y_train != state.y[labelled]))
    else:
        y_out = state.y

    return ClusteringState(
        Z=state.Z,
        y=y_out,
        masks=new_masks,
        entropy=entropy,
        confidence=confidence,
        metadata=state.metadata,
        timestamp=state.timestamp,
        random_seed=state.random_seed
    )
