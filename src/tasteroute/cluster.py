"""k-means clustering of places by location, for multi-day plans.

A 2-3 day plan should not zig-zag across the city every day: group the
recommended places into k neighborhoods and give each day one group.

k-means minimizes the within-cluster sum of squared distances
    sum_c sum_{x in c} ||x - mu_c||^2
by alternating two steps until no point changes cluster (Lloyd's algorithm):
    assign   each point to its nearest center
    update   each center to the mean of its points
It only finds a local optimum, so it is restarted n_init times from k-means++
starting centers (each new center drawn with probability proportional to the
squared distance to the nearest center chosen so far) and the best run is kept.

Plain k-means can put 11 places in one day and 1 in the other, so
balanced_kmeans adds a capacity: the assign step fills clusters greedily, the
closest (point, center) pairs first, until each cluster holds `size` points.
"""
import numpy as np

from .roads import KM_PER_DEG_LAT


def to_km(lat, lon):
    """Equirectangular projection to km, good enough within a city."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    # np.column_stack([a, b]) puts the two 1-D arrays side by side: an n x 2 matrix of (north km, east km)
    return np.column_stack([lat * KM_PER_DEG_LAT, lon * KM_PER_DEG_LAT * np.cos(np.radians(lat.mean()))])


def _plus_plus(X, k, rng):
    """k-means++ starting centers."""
    # rng.integers(n): one random index in [0, n), the first center
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(axis=2), axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
        # rng.choice(n, p=p): one index in [0, n) drawn with probabilities p
        centers.append(X[rng.choice(len(X), p=p)])
    return np.array(centers)


def kmeans(X, k, n_init=10, max_iter=100, seed=42):
    """Cluster the rows of X into k groups.

    Returns:
        (labels, centers, inertia): cluster index per row, the k centers, and
        the within-cluster sum of squared distances of the best run.
    """
    X = np.asarray(X, dtype=float)
    k = min(k, len(X))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = _plus_plus(X, k, rng)
        labels = np.full(len(X), -1)
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - centers[None]) ** 2).sum(axis=2)
            new = d2.argmin(axis=1)
            if np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                if np.any(labels == c):  # an empty cluster keeps its old center
                    centers[c] = X[labels == c].mean(axis=0)
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (labels.copy(), centers.copy(), inertia)
    return best


def balanced_kmeans(X, k, size, max_iter=50, seed=42):
    """k clusters of at most `size` points each (k * size >= len(X)).

    Starts from the k-means centers, then alternates a capacity-constrained
    assignment (all point-center pairs sorted by distance, each point taken by
    the nearest center that still has room) with the usual center update.

    Returns:
        (labels, centers): cluster index per row and the k centers.
    """
    X = np.asarray(X, dtype=float)
    k = min(k, len(X))
    if k * size < len(X):
        raise ValueError("k * size must be at least the number of points")
    _, centers, _ = kmeans(X, k, seed=seed)
    labels = np.full(len(X), -1)
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centers[None]) ** 2).sum(axis=2)
        new, room = np.full(len(X), -1), np.full(k, size)
        # np.argsort(d2, axis=None) sorts the whole n x k matrix as one flat list and
        # returns flat positions; divmod(position, k) turns each back into (point, center)
        for flat in np.argsort(d2, axis=None, kind="stable"):
            p, c = divmod(int(flat), k)
            if new[p] < 0 and room[c] > 0:
                new[p], room[c] = c, room[c] - 1
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            if np.any(labels == c):
                centers[c] = X[labels == c].mean(axis=0)
    return labels, centers
