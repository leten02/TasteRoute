import numpy as np
import pandas as pd

from tasteroute.models.mf import MF


def _toy(seed=0, n_users=30, n_items=20):
    # ratings generated from a rank-1 taste structure plus biases
    rng = np.random.default_rng(seed)
    a, b = rng.normal(0, 1, n_users), rng.normal(0, 1, n_items)
    u, i = np.meshgrid(np.arange(n_users), np.arange(n_items), indexing="ij")
    u, i = u.ravel(), i.ravel()
    stars = np.clip(np.round(3.5 + 0.6 * a[u] * b[i]), 1, 5)
    df = pd.DataFrame({"u": u, "i": i, "stars": stars})
    mask = rng.random(len(df)) < 0.8
    return df[mask], df[~mask], n_users, n_items


def test_mf_training_error_falls():
    train, val, nu, ni = _toy()
    m = MF(nu, ni, f=2, eta=0.02, lam=0.01, max_epochs=30).fit(train)
    curve = [h["train_rmse"] for h in m.history]
    assert curve[-1] < curve[0]


def test_mf_early_stopping_restores_best_epoch():
    train, val, nu, ni = _toy()
    m = MF(nu, ni, f=2, eta=0.02, lam=0.01, max_epochs=200, patience=3).fit(train, val)
    best_val = min(h["val_rmse"] for h in m.history)
    final = np.sqrt(np.mean((val.stars.values - m.predict(val.u.values, val.i.values)) ** 2))
    assert np.isclose(final, best_val)
    assert len(m.history) <= 200


def test_mf_predictions_in_star_range():
    train, _, nu, ni = _toy()
    m = MF(nu, ni, max_epochs=5).fit(train)
    s = m.score_all(0)
    assert s.shape == (ni,) and s.min() >= 1 and s.max() <= 5


def test_single_pair_prediction_matches_batch():
    m = MF(3, 3, f=2, max_epochs=1).fit(pd.DataFrame({"u": [0, 1, 2], "i": [0, 1, 2], "stars": [5.0, 3.0, 1.0]}))
    assert np.isclose(m.predict(1, 2), m.predict(np.array([1]), np.array([2]))[0])
