import numpy as np
import pandas as pd

from tasteroute.models.hybrid import Hybrid


class _Const:
    def __init__(self, value, n_users=3, n_items=2):
        self.value, self.n_users, self.n_items = value, n_users, n_items

    def predict(self, u, i):
        return np.full(len(u), self.value)

    def score_all(self, u):
        return np.full(self.n_items, self.value)


def test_fixed_alpha_blend():
    h = Hybrid(_Const(4.0), _Const(2.0), alpha=0.25).fit(pd.DataFrame({"u": [0], "i": [0]}))
    assert np.allclose(h.predict([0, 1], [0, 1]), 0.25 * 4 + 0.75 * 2)


def test_adaptive_alpha_matches_worked_example():
    train = pd.DataFrame({"u": [0] * 45 + [1], "i": [0] * 46})
    h = Hybrid(_Const(4.0), _Const(2.0), kappa=100).fit(train)
    assert np.isclose(h.weight(np.array([0]))[0], 45 / 145)  # proposal 9.4: 0.31
    assert h.weight(np.array([2]))[0] == 0  # no ratings -> CBF only


def test_kappa_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        Hybrid(None, None, kappa=0)


def test_alpha_must_be_a_weight():
    import pytest

    with pytest.raises(ValueError):
        Hybrid(None, None, alpha=2)
    Hybrid(None, None, alpha=2, kappa=10)  # alpha is ignored when kappa is set
