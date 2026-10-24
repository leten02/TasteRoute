import math

import pandas as pd
import pytest

from tasteroute.metrics import mae, rmse
from tasteroute.models.baseline import BiasModel, GlobalMean


def test_rmse_mae_by_hand():
    assert math.isclose(rmse([1, 2, 3], [1, 2, 5]), math.sqrt(4 / 3))
    assert math.isclose(mae([1, 2, 3], [1, 2, 5]), 2 / 3)


def test_rmse_mae_reject_mismatched_or_empty_input():
    with pytest.raises(ValueError):
        rmse([1, 2], [[1], [2]])  # would broadcast to 2 x 2
    with pytest.raises(ValueError):
        mae([], [])


def test_bias_model_learns_harsh_user_and_good_item():
    # user 0 always rates 1 below user 1; item 1 always gets 1 more than item 0
    train = pd.DataFrame({"u": [0, 0, 1, 1], "i": [0, 1, 0, 1], "stars": [2, 3, 3, 4]})
    m = BiasModel(2, 3, lam=0.0, n_iter=50).fit(train)
    assert m.b_u[0] < m.b_u[1]
    assert m.b_i[0] < m.b_i[1]
    assert m.b_i[2] == 0  # item never seen in training keeps a zero bias
    pred = m.predict(train.u.values, train.i.values)
    assert rmse(train.stars, pred) < 1e-6


def test_global_mean():
    train = pd.DataFrame({"u": [0, 1], "i": [0, 0], "stars": [2, 4]})
    m = GlobalMean(2, 1).fit(train)
    assert list(m.predict([0, 1], [0, 0])) == [3.0, 3.0]
