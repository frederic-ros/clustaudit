import numpy as np
import pytest
from scipy.spatial import cKDTree

from clustaudit import generate_dataset, NoiseParams, AmbiguityParams, StructuralParams
from clustaudit.metrics import profile_from_state


def _gen(**noise):
    return generate_dataset(600, 10, 4, noise=NoiseParams(**noise), random_seed=7)


def test_seed_reproducible():
    a, b = _gen(noise_ratio=0.1), _gen(noise_ratio=0.1)
    assert np.array_equal(a.Z, b.Z) and np.array_equal(a.y, b.y)


def test_eta_int_changes_the_data():
    """Regression test for the v1 bug: five eta_int levels produced identical datasets."""
    zs = [_gen(noise_ratio=0.2, internal_noise_ratio=e).Z for e in (0.0, 0.5, 1.0)]
    assert not np.array_equal(zs[0], zs[1])
    assert not np.array_equal(zs[1], zs[2])


def test_internal_noise_is_between_clusters():
    """Bridge noise must be far closer to the clusters than distant noise."""
    st_int = _gen(noise_ratio=0.2, internal_noise_ratio=1.0)
    st_dist = _gen(noise_ratio=0.2, internal_noise_ratio=0.0)
    for st in (st_int, st_dist):
        assert (st.y == -1).sum() == 120  # n_core = 480, noise = 480 * 0.2 / 0.8 = 120
    tree = cKDTree(st_int.Z[st_int.y >= 0])
    d_int = np.median(tree.query(st_int.Z[st_int.y == -1])[0])
    tree = cKDTree(st_dist.Z[st_dist.y >= 0])
    d_dist = np.median(tree.query(st_dist.Z[st_dist.y == -1])[0])
    assert d_int < 0.5 * d_dist


def test_noise_split_masks_are_populated():
    """Regression test for v1: the distant_noise mask was always empty."""
    st = _gen(noise_ratio=0.2, internal_noise_ratio=0.5)
    n_int, n_dist = st.masks['internal_noise'].sum(), st.masks['distant_noise'].sum()
    assert n_int + n_dist == (st.y == -1).sum()
    assert n_int > 0 and n_dist > 0
    # internal noise must be the majority of what was placed as bridge noise, distant of the rest
    assert n_dist >= 0.6 * 60 and n_int >= 0.6 * 60


def test_profile_sees_noise_geometry():
    """D_dataset and the k-NN profile must respond to eta_int at fixed eta."""
    p0 = profile_from_state(_gen(noise_ratio=0.2, internal_noise_ratio=0.0))
    p1 = profile_from_state(_gen(noise_ratio=0.2, internal_noise_ratio=1.0))
    assert p1.components['noise'] > p0.components['noise']
    assert p1.knn['noise_proximity'] > p0.knn['noise_proximity']


def test_masks_partition_the_points():
    st = _gen(noise_ratio=0.15)
    m = st.masks
    lab = st.y >= 0
    assert np.array_equal(m['core'] | m['boundary'], lab)
    assert not np.any(m['core'] & m['boundary'])
    assert np.array_equal(m['internal_noise'] | m['distant_noise'], ~lab)


def test_legacy_bounding_box_mode_still_runs():
    st = generate_dataset(300, 5, 3, noise=NoiseParams(noise_ratio=0.1, internal_noise_mode="bounding_box"),
                          random_seed=1)
    assert (st.y == -1).sum() > 0


def test_in_sample_confidence_option():
    st = generate_dataset(300, 5, 3, ambiguity=AmbiguityParams(confidence_mode="in_sample"), random_seed=1)
    assert st.confidence.shape == (300,)


def test_density_parameters_change_the_data():
    """Regression test: v1 gated both density mechanisms behind a flag that was never set."""
    from clustaudit import DensityParams
    import numpy as np
    def gen(dp):
        return generate_dataset(600, 10, 4, structural=StructuralParams(density_params=dp, enable_anisotropy=False),
                                noise=NoiseParams(noise_ratio=0.0), random_seed=5)
    base = gen(DensityParams())
    inter = gen(DensityParams(inter_density_variation=1.0))
    intra = gen(DensityParams(intra_density_mode="core_dense", intra_density_strength=3.0))
    assert not np.array_equal(base.Z, inter.Z)
    assert not np.array_equal(base.Z, intra.Z)
    # no duplicated points after intra-density shaping
    assert len(np.unique(intra.Z.round(9), axis=0)) == len(intra.Z)
    # core_dense: median radius shrinks
    def med_r(st):
        return np.median([np.linalg.norm(st.Z[st.y == k] - st.Z[st.y == k].mean(0), axis=1).mean() for k in range(4)])
    assert med_r(intra) < 0.8 * med_r(base)
