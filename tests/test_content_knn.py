import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tasteroute.models.content_knn import ContentKNN


def _setup():
    # items 0, 1 look alike (feature a); items 2, 3 look alike (feature b)
    X = sparse.csr_matrix(np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float))
    # many users give everything 3 stars; user 0 loves item 0 and dislikes item 2
    rows = [(u, i, 3.0) for u in range(1, 30) for i in range(4)] + [(0, 0, 5.0), (0, 2, 1.0)]
    return X, pd.DataFrame(rows, columns=["u", "i", "stars"])


def test_one_rating_transfers_to_similar_places():
    X, train = _setup()
    m = ContentKNN(30, 4, X, shrink=0.0).fit(train)
    p = m.predict(np.array([0, 0]), np.array([1, 3]))
    assert p[0] > p[1]  # item 1 looks like the loved item 0, item 3 like the disliked item 2
    assert np.allclose(m.score_all(0)[[1, 3]], p)


def test_shrink_pulls_toward_bias():
    X, train = _setup()
    loose = ContentKNN(30, 4, X, shrink=0.0).fit(train).predict(np.array([0]), np.array([1]))[0]
    tight = ContentKNN(30, 4, X, shrink=100.0).fit(train)
    base = tight.bias.baseline(np.array([0]), np.array([1]))[0]
    assert abs(tight.predict(np.array([0]), np.array([1]))[0] - base) < abs(loose - base)


def test_user_without_ratings_gets_bias_and_negative_shrink_rejected():
    X, train = _setup()
    m = ContentKNN(31, 4, X).fit(train)  # user 30 has no ratings
    assert np.allclose(m.predict(np.array([30]), np.array([2])), m.bias.baseline(np.array([30]), np.array([2])))
    with pytest.raises(ValueError):
        ContentKNN(30, 4, X, shrink=-1)
