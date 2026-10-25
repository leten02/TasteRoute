"""Content-based neighbors: predict from the user's ratings of similar-looking places.

Per-user ridge CBF (cbf.py) has to fit a taste vector from a handful of
ratings. This model needs no per-user fit: similarity between places comes
from their content features (categories and attributes), so one rating already
says something about every place that looks like it.

    r_hat_ui = b_ui + sum_j s_ij (r_uj - b_uj) / (sum_j |s_ij| + shrink)

j runs over the places u rated, s_ij is the cosine similarity of the content
vectors of i and j, and shrink pulls the correction toward 0 (the bias
baseline) when the user's places are not similar to i or there are few of them.
"""
import numpy as np

from .baseline import MAX_STARS, MIN_STARS, BiasModel


class ContentKNN:
    """Item-content kNN on top of the bias baseline.

    Args:
        n_users, n_items: array sizes.
        X: item x feature matrix (scipy sparse), e.g. item_extras.combine_features.
        shrink: added to the similarity sum in the denominator (>= 0).
        bias_lam: lambda of the bias model.
    """

    def __init__(self, n_users, n_items, X, shrink=1.0, bias_lam=5.0):
        if shrink < 0:
            raise ValueError("shrink must be >= 0")
        self.n_users, self.n_items = n_users, n_items
        X = X.tocsr().astype(float)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        # row-normalize once, so X_i . X_j is the cosine similarity
        self.Xn = X.multiply(1.0 / norms[:, None]).tocsr()
        self.shrink, self.bias_lam = shrink, bias_lam

    def fit(self, train):
        """Store each user's rated places and their residuals r - b (returns self)."""
        self.bias = BiasModel(self.n_users, self.n_items, lam=self.bias_lam).fit(train)
        u, i = train.u.values, train.i.values
        res = train.stars.values - self.bias.baseline(u, i)
        order = np.argsort(u, kind="stable")
        starts = np.searchsorted(u[order], np.arange(self.n_users + 1))
        i_sorted, res_sorted = i[order], res[order]
        self.rated = [i_sorted[starts[k]:starts[k + 1]] for k in range(self.n_users)]
        self.res = [res_sorted[starts[k]:starts[k + 1]] for k in range(self.n_users)]
        return self

    def _correction(self, user, items):
        """sum_j s_ij res_uj / (sum_j |s_ij| + shrink) for these items."""
        rated = self.rated[user]
        if len(rated) == 0:
            return np.zeros(len(items))
        S = (self.Xn[items] @ self.Xn[rated].T).toarray()  # len(items) x len(rated) cosines
        return S @ self.res[user] / (np.abs(S).sum(axis=1) + self.shrink + 1e-12)

    def predict(self, u, i):
        """Predicted stars, clipped to 1-5, for each pair (u[k], i[k])."""
        u, i = np.asarray(u), np.asarray(i)
        out = self.bias.baseline(u, i).astype(float)
        if len(u) == 0:
            return out
        order = np.argsort(u, kind="stable")
        bounds = np.flatnonzero(np.diff(u[order])) + 1
        for idx in np.split(order, bounds):
            out[idx] += self._correction(u[idx[0]], i[idx])
        return np.clip(out, MIN_STARS, MAX_STARS)

    def score_all(self, u):
        """Predicted stars for user u on every item."""
        base = self.bias.mu + self.bias.b_u[u] + self.bias.b_i
        return np.clip(base + self._correction(u, np.arange(self.n_items)), MIN_STARS, MAX_STARS)
