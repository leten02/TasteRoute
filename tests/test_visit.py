import numpy as np
import pandas as pd

from tasteroute.geo import user_anchors
from tasteroute.models.visit import LogisticRegression, VisitFeatures, rank_learned, sigmoid


def test_sigmoid_is_stable_at_extremes():
    s = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.allclose(s, [0.0, 0.5, 1.0]) and np.all(np.isfinite(s))


def test_logistic_regression_recovers_known_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 2)) * [1.0, 3.0] + [0.5, -2.0]
    true_w, true_b = np.array([1.5, -0.8]), 0.3
    y = (rng.random(20000) < sigmoid(X @ true_w + true_b)).astype(float)
    m = LogisticRegression(lam=0).fit(X, y)
    w, b = m.coef()
    assert np.allclose(w, true_w, atol=0.1) and abs(b - true_b) < 0.2
    assert all(a >= c - 1e-12 for a, c in zip(m.history, m.history[1:]))  # loss never goes up


def test_converges_on_rare_positives():
    # like visits: about 1 positive in 200 rows
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40000, 2))
    y = (rng.random(40000) < sigmoid(X @ np.array([-1.2, 0.7]) - 6.0)).astype(float)
    m = LogisticRegression(lam=0).fit(X, y)
    assert len(m.history) < 500
    assert np.allclose(m.coef()[0], [-1.2, 0.7], atol=0.25)


def test_l2_shrinks_weights():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    y = (X[:, 0] + rng.normal(size=2000) > 0).astype(float)
    free = np.abs(LogisticRegression(lam=0).fit(X, y).w).sum()
    shrunk = np.abs(LogisticRegression(lam=1.0).fit(X, y).w).sum()
    assert shrunk < free


def test_to_dict_round_trip():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(500, 2)), rng.integers(0, 2, 500)
    m = LogisticRegression(lam=0.01).fit(X, y)
    m2 = LogisticRegression.from_dict(m.to_dict())
    assert np.allclose(m.decision_function(X), m2.decision_function(X))


def _toy():
    # 3 places on a north-south line, user 0 rated places 0 and 1
    items = pd.DataFrame({"i": range(3), "lat": [40.0, 40.01, 40.1], "lon": [-75.0] * 3})
    train = pd.DataFrame({"u": [0, 0, 1], "i": [0, 1, 2]})
    anchors = user_anchors(train, items, 2)
    return items, train, VisitFeatures(items, train, anchors, np.array([10.0, 1.0, 0.0]))


def test_features_and_radius():
    items, train, f = _toy()
    X = f.build(0, np.array([0, 2]), np.array([4.0, 3.0]))
    assert X.shape == (2, 5)
    assert X[0, 0] < X[1, 0]  # place 0 is nearer the anchor than place 2
    assert np.isclose(X[0, 4], np.log1p(10.0)) and X[1, 3] == 3.0
    assert f.radius[1] == 0.5  # one training place: radius 0, raised to the floor


def test_rank_learned_prefers_near_places_when_distance_weight_is_negative():
    items, train, f = _toy()
    m = LogisticRegression.from_dict({"lam": 0, "mean": [0] * 5, "std": [1] * 5,
                                      "w": [-2.0, 0.0, 0.0, 0.0, 0.0], "b": 0.0})
    cand = np.array([2, 0, 1])
    X = f.build(0, cand, np.full(3, 4.0))
    assert list(rank_learned(m, X, cand, k=2)) == [0, 1]  # place 0 is nearest the anchor
