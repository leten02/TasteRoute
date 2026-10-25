import numpy as np
import pandas as pd

from tasteroute.models.baseline import BiasModel
from tasteroute.models.item_knn import ItemKNN


def _toy():
    # items 0 and 1 are rated identically by everyone; item 2 is unrelated
    rows = []
    for u, (a, c) in enumerate([(5, 1), (1, 5), (5, 2), (2, 4), (4, 1)]):
        rows += [(u, 0, a), (u, 1, a), (u, 2, c)]
    rows.append((5, 0, 5))  # user 5 rated only item 0
    return pd.DataFrame(rows, columns=["u", "i", "stars"])


def test_similar_items_get_high_similarity():
    m = ItemKNN(6, 3, "cosine", k=2, beta=0).fit(_toy())
    assert m.S[0, 1] > m.S[0, 2]
    assert m.S[0, 0] == 0  # an item is never its own neighbor


def test_neighbor_moves_prediction_toward_neighbor_rating():
    df = _toy()
    m = ItemKNN(6, 3, "cosine", k=1, beta=0).fit(df)
    b = BiasModel(6, 3, lam=5.0).fit(df)
    # user 5 loved item 0, its twin item 1 should be predicted above the bias baseline
    assert m.predict([5], [1])[0] > b.predict([5], [1])[0]


def test_user_without_ratings_falls_back_to_bias():
    df = _toy()
    m = ItemKNN(7, 3, "pearson", k=2).fit(df)
    b = BiasModel(7, 3, lam=5.0).fit(df)
    assert np.allclose(m.predict([6, 6], [0, 2]), b.predict([6, 6], [0, 2]))


def test_score_all_shape_and_range():
    m = ItemKNN(6, 3, "adjusted", k=2).fit(_toy())
    s = m.score_all(0)
    assert s.shape == (3,) and s.min() >= 1 and s.max() <= 5


def test_empty_batch_predicts_nothing():
    m = ItemKNN(3, 3, "cosine", k=2, beta=0).fit(pd.DataFrame({"u": [0, 1, 2], "i": [0, 1, 2], "stars": [5.0, 3.0, 1.0]}))
    assert m.predict(np.array([], dtype=int), np.array([], dtype=int)).shape == (0,)


def test_rated_item_is_not_its_own_neighbor():
    # users who like item 0 dislike item 1: adjusted cosine s_01 = -1. Scoring the
    # rated item 0 with k = 1 must use item 1, not item 0 itself (similarity 0),
    # which would silently fall back to the bias baseline.
    df = pd.DataFrame({"u": [0, 0, 1, 1, 2, 2, 3, 3], "i": [0, 1, 0, 1, 0, 1, 2, 0],
                       "stars": [5.0, 1.0, 5.0, 1.0, 4.0, 2.0, 3.0, 3.0]})
    m = ItemKNN(4, 3, "adjusted", k=1, beta=0).fit(df)
    assert m.S[0, 1] < 0
    base = np.clip(m.bias.baseline(np.array([0]), np.array([0]))[0], 1, 5)
    assert not np.isclose(m.score_all(0)[0], base)
