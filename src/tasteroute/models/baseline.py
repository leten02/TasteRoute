"""Baselines: global mean and the bias model.

Every other recommender (MF, item-kNN, CBF) is built on top of BiasModel, so its
prediction b_ui = mu + b_u + b_i is the shared starting point.

Proposal section 5.1.
"""
import numpy as np

MIN_STARS, MAX_STARS = 1.0, 5.0


def _train_mean(train):
    """Mean training rating; fails loudly on empty or non-finite stars (else NaN spreads)."""
    stars = train.stars.values.astype(float)
    if len(stars) == 0 or not np.all(np.isfinite(stars)):
        raise ValueError("training ratings must be non-empty and finite")
    return float(stars.mean())


def _safe_div(num, den):
    """num / den, with 0 where den == 0 (an id with no ratings and lam = 0).

    np.divide(..., out=zeros, where=mask) only divides where mask is True and
    leaves the preset zeros elsewhere.
    """
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


class GlobalMean:
    """Predict the training mean rating mu for every (user, item) pair."""

    def __init__(self, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items

    def fit(self, train):
        """Learn the parameters from training ratings (columns u, i, stars); returns self."""
        self.mu = _train_mean(train)
        return self

    def predict(self, u, i):
        """Predicted stars, clipped to 1-5, for each pair (u[k], i[k])."""
        return np.full(len(u), self.mu)

    def score_all(self, u):
        """Predicted stars for user u on every item (array of length n_items)."""
        return np.full(self.n_items, self.mu)


class BiasModel:
    """r_hat = mu + b_u + b_i, fitted by alternating closed-form updates.

    Loss: sum (r - mu - b_u - b_i)^2 + lam * (sum b_u^2 + sum b_i^2).
    Holding b_u fixed, the best b_i is the regularized mean of its residuals,
    and the other way round:

        b_i = sum_{u in U_i} (r_ui - mu - b_u) / (lam + |U_i|)
        b_u = sum_{i in I_u} (r_ui - mu - b_i) / (lam + |I_u|)

    lam pulls the bias of a user or item with few ratings toward 0.

    Args:
        n_users, n_items: array sizes (see data.n_users_items).
        lam: L2 regularization strength.
        n_iter: number of alternating rounds; 10 is enough to converge here.
    """

    def __init__(self, n_users, n_items, lam=5.0, n_iter=10):
        self.n_users, self.n_items = n_users, n_items
        self.lam, self.n_iter = lam, n_iter

    def fit(self, train):
        """Learn the parameters from training ratings (columns u, i, stars); returns self."""
        u, i = train.u.values, train.i.values
        self.mu = _train_mean(train)
        y = train.stars.values - self.mu
        # np.bincount(x, weights=w, minlength=n) sums w per integer id in x and
        # returns an array of length n; without weights it counts occurrences.
        n_u = np.bincount(u, minlength=self.n_users)
        n_i = np.bincount(i, minlength=self.n_items)
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        for _ in range(self.n_iter):
            self.b_i = _safe_div(np.bincount(i, weights=y - self.b_u[u], minlength=self.n_items), self.lam + n_i)
            self.b_u = _safe_div(np.bincount(u, weights=y - self.b_i[i], minlength=self.n_users), self.lam + n_u)
        return self

    def baseline(self, u, i):
        """Unclipped mu + b_u + b_i; the base term other models add to."""
        return self.mu + self.b_u[u] + self.b_i[i]

    def predict(self, u, i):
        """Predicted stars, clipped to 1-5, for each pair (u[k], i[k])."""
        return np.clip(self.baseline(u, i), MIN_STARS, MAX_STARS)

    def score_all(self, u):
        """Predicted stars for user u on every item (array of length n_items)."""
        return np.clip(self.mu + self.b_u[u] + self.b_i, MIN_STARS, MAX_STARS)
