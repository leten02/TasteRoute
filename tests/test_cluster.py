import numpy as np

from tasteroute.cluster import kmeans, to_km


def test_kmeans_finds_separated_groups():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0], 0.1, (20, 2))
    b = rng.normal([5, 5], 0.1, (20, 2))
    labels, centers, inertia = kmeans(np.vstack([a, b]), 2)
    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1 and labels[0] != labels[20]
    assert inertia < 2.0


def test_kmeans_more_clusters_than_points_and_to_km():
    labels, centers, _ = kmeans(np.array([[0.0, 0.0], [1.0, 1.0]]), 5)
    assert sorted(labels) == [0, 1] and len(centers) == 2
    X = to_km([40.0, 40.01], [-75.0, -75.0])
    assert np.isclose(X[1, 0] - X[0, 0], 1.1132, atol=1e-3)


def test_balanced_kmeans_caps_cluster_size():
    import pytest

    from tasteroute.cluster import balanced_kmeans

    # 11 points in one tight group and 1 far away: plain k-means splits 11 / 1
    X = np.vstack([np.random.default_rng(0).normal(0, 0.2, (11, 2)), [[10.0, 10.0]]])
    plain, _, _ = kmeans(X, 2)
    assert sorted(np.bincount(plain)) == [1, 11]
    labels, _ = balanced_kmeans(X, 2, size=6)
    assert sorted(np.bincount(labels)) == [6, 6]
    with pytest.raises(ValueError):
        balanced_kmeans(X, 2, size=5)
