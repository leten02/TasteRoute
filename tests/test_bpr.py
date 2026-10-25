import numpy as np
import pandas as pd

from tasteroute.models.bpr import BPR


def _clusters(seed=0):
    """Two user groups that each go to their own half of the items."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(40):
        group = range(0, 10) if u < 20 else range(10, 20)
        for i in rng.choice(list(group), size=6, replace=False):
            rows.append((u, int(i)))
    return pd.DataFrame(rows, columns=["u", "i"])


def test_bpr_learns_group_preferences():
    train = _clusters()
    m = BPR(40, 20, f=4, lr=0.1, lam=1e-4, batch=32, max_epochs=40).fit(train)
    s0, s1 = m.score_all(0), m.score_all(39)
    assert s0[:10].mean() > s0[10:].mean()  # group A user prefers group A items
    assert s1[10:].mean() > s1[:10].mean()
    assert m.history[-1]["loss"] < m.history[0]["loss"]


def test_negatives_are_never_positives():
    train = _clusters()
    m = BPR(40, 20, f=2, max_epochs=1).fit(train)
    u = np.repeat(np.arange(40), 50)
    j = m._negatives(u, np.random.default_rng(1))
    seen = set(zip(train.u, train.i))
    assert not any((a, b) in seen for a, b in zip(u.tolist(), j.tolist()))


def test_early_stopping_restores_best_epoch():
    train = _clusters()
    scores = iter([0.1, 0.5, 0.4, 0.3, 0.2, 0.1])
    m = BPR(40, 20, f=2, max_epochs=6, patience=3).fit(train, val_score=lambda _: next(scores))
    assert m.best_epoch == 2 and len(m.history) == 5


def test_new_visitor_and_save_load(tmp_path):
    m = BPR(40, 20, f=2, max_epochs=2).fit(_clusters())
    assert np.allclose(m.score_all(None), m.b)
    m.save(tmp_path / "bpr.npz")
    m2 = BPR.load(tmp_path / "bpr.npz")
    assert np.allclose(m.score_all(3), m2.score_all(3))
