import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from clustaudit import generate_dataset, NoiseParams, StructuralParams, difficulty_profile
from clustaudit.metrics import profile_from_state, knn_profile
from clustaudit.audit import evaluate_partition, sweep, audit_dataset, run_pool


def test_profile_from_external_data_matches_internal_chain():
    """The profile of (X, y) must equal the profile of the STATE that produced them."""
    st = generate_dataset(500, 8, 4, noise=NoiseParams(noise_ratio=0.1), random_seed=3)
    p_int = profile_from_state(st, seed=3)
    p_ext = difficulty_profile(st.Z, st.y, seed=3)
    assert abs(p_int.score - p_ext.score) < 1e-9
    for k in p_int.components:
        assert abs(p_int.components[k] - p_ext.components[k]) < 1e-9


def test_profile_orders_easy_before_hard():
    easy = make_blobs(600, centers=4, n_features=6, cluster_std=0.4, random_state=0)
    hard = make_blobs(600, centers=4, n_features=6, cluster_std=3.0, random_state=0)
    pe, ph = difficulty_profile(*easy), difficulty_profile(*hard)
    assert pe.score < ph.score
    assert pe.knn['edge_impurity'] < ph.knn['edge_impurity']
    assert pe.baselines['knn_cv_error'] < ph.baselines['knn_cv_error']


def test_knn_profile_fragmentation_detects_split_classes():
    X, y = make_blobs(400, centers=4, n_features=5, cluster_std=0.3, random_state=1)
    merged = y // 2  # two well separated blobs per "class" -> fragmented classes
    assert knn_profile(X, merged)['fragmentation'] > 0.3
    assert knn_profile(X, y)['fragmentation'] < 0.05


def test_evaluation_conventions():
    y = np.array([0, 0, 0, 1, 1, 1, -1, -1])
    X = np.vstack([np.random.RandomState(0).normal(size=(8, 2))])
    # perfect on inliers, rejects the true noise
    lab = np.array([0, 0, 0, 1, 1, 1, -1, -1])
    m = evaluate_partition(X, y, lab)
    assert m['ARI'] == 1.0 and m['ARI_full'] == 1.0 and m['noise_recall'] == 1.0
    # rejects a true inlier: inlier ARI drops, 'kept' convention does not see it
    lab2 = np.array([0, 0, -1, 1, 1, 1, -1, -1])
    m2 = evaluate_partition(X, y, lab2)
    assert m2['ARI'] < 1.0 and m2['ARI_kept'] == 1.0
    assert m2['n_clusters_found'] == 2 and not m2['degenerate']


def test_sweep_runs_end_to_end():
    algo = lambda X, seed: KMeans(5, random_state=seed, n_init=3).fit_predict(X)
    df = sweep(algo, factor="eta_int", values=[0.0, 1.0], seeds=[1], n_samples=300, dimension=5)
    assert len(df) == 2 and {"ARI", "D_dataset", "knn_edge_impurity", "S_noise"} <= set(df.columns)
    assert df["value"].tolist() == [0.0, 1.0]
    assert df["S_noise"].iloc[1] > df["S_noise"].iloc[0]


def test_run_pool_fixed_protocol():
    st = generate_dataset(300, 5, 3, random_seed=2)
    out = run_pool(st.Z, 2, names=["KMeans", "DBSCAN"], criterion="fixed", n_clusters=3)
    assert set(out) == {"KMeans", "DBSCAN"}
    assert len(np.unique(out["KMeans"])) == 3


def test_min_cluster_recall_and_failure_mode():
    y = np.array([0]*8 + [1]*8 + [2]*2)          # cluster 2 is the rare one
    X = np.random.RandomState(0).normal(size=(18, 2))
    lab_good = y.copy()
    lab_merge = np.array([0]*8 + [1]*10)         # rare cluster absorbed by cluster 1
    m = evaluate_partition(X, y, lab_good); assert m["min_cluster_recall"] == 1.0 and m["failure_mode"] == "ok"
    m2 = evaluate_partition(X, y, lab_merge)
    assert m2["min_cluster_recall"] == 1.0            # fully captured...
    assert abs(m2["min_cluster_f1"] - 2*0.2*1/(1.2)) < 1e-9   # ...but merged: F1 sees it
    assert m2["ARI"] > 0.5
    lab_one = np.zeros(18, int); m3 = evaluate_partition(X, y, lab_one); assert m3["failure_mode"] == "declared"
    lab_rand = np.random.RandomState(1).randint(0, 3, 18); m4 = evaluate_partition(X, y, lab_rand)
    assert m4["failure_mode"] in ("silent", "ok")


def test_score_knn_in_profile():
    from sklearn.datasets import make_blobs
    p = difficulty_profile(*make_blobs(300, centers=3, n_features=4, cluster_std=2.0, random_state=0))
    assert 0 <= p.score_knn <= 1 and "D_knn" in p.to_dict()
