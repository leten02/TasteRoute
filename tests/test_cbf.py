import numpy as np
import pandas as pd
from scipy import sparse

from tasteroute.models.cbf import CBF


def _toy():
    # 4 items, 2 categories: items 0,1 are "A", items 2,3 are "B"
    X = sparse.csr_matrix(np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float))
    # user 0 loves A and dislikes B; user 1 is neutral
    df = pd.DataFrame({"u": [0, 0, 1, 1, 1], "i": [0, 2, 0, 2, 3], "stars": [5, 1, 3, 3, 3]})
    return X, df


def test_learns_category_taste():
    X, df = _toy()
    m = CBF(2, 4, X, lam=0.1).fit(df)
    assert m.theta[0, 0] > 0 > m.theta[0, 1]
    # unseen item 1 (category A) is predicted above unseen-by-u0 item 3 (category B)
    p = m.predict([0, 0], [1, 3])
    assert p[0] > p[1]


def test_dual_and_primal_ridge_agree():
    rng = np.random.default_rng(0)
    Xu, y, lam = rng.normal(size=(3, 6)), rng.normal(size=3), 2.0
    primal = np.linalg.solve(Xu.T @ Xu + lam * np.eye(6), Xu.T @ y)
    dual = Xu.T @ np.linalg.solve(Xu @ Xu.T + lam * np.eye(3), y)
    assert np.allclose(primal, dual)


def test_score_all_matches_predict():
    X, df = _toy()
    m = CBF(2, 4, X, lam=1.0).fit(df)
    assert np.allclose(m.score_all(0), m.predict([0] * 4, [0, 1, 2, 3]))
