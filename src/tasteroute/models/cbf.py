"""Content-based filtering: per-user ridge regression on IDF-weighted categories.

Each user gets a linear taste vector theta_u over the category features x_i. It
explains what the bias baseline leaves over:

    y_ui = r_ui - b_ui              r_hat_ui = b_ui + x_i . theta_u
    theta_u = (X_u^T X_u + lam I)^-1 X_u^T y_u

X_u holds the feature rows of the places u rated. Fitting on residuals of b_ui,
not of the user's mean, matters for new users: with one rating the user mean is
that rating, the residual is 0 and nothing is learned (the proposal's first run
reached RMSE 1.50 that way).

Proposal section 5.4.
"""
import numpy as np

from .baseline import MAX_STARS, MIN_STARS, BiasModel


class CBF:
    """Per-user ridge CBF.

    Args:
        n_users, n_items: array sizes (see data.n_users_items).
        X: item x feature matrix (scipy sparse, from data.load_item_features).
        lam: ridge strength lambda_c; larger keeps theta_u closer to 0.
        bias_lam: lambda of the bias model that provides b_ui.
    """

    def __init__(self, n_users, n_items, X, lam=10.0, bias_lam=5.0):
        self.n_users, self.n_items = n_users, n_items
        self.X = X.tocsr()
        self.lam, self.bias_lam = lam, bias_lam

    def fit(self, train):
        """Learn the parameters from training ratings (columns u, i, stars); returns self."""
        u, i = train.u.values, train.i.values
        self.bias = BiasModel(self.n_users, self.n_items, lam=self.bias_lam).fit(train)
        y = train.stars.values - self.bias.baseline(u, i)

        d = self.X.shape[1]
        self.theta = np.zeros((self.n_users, d))
        order = np.argsort(u, kind="stable")
        starts = np.searchsorted(u[order], np.arange(self.n_users + 1))
        for user in range(self.n_users):
            rows = order[starts[user]:starts[user + 1]]
            if len(rows) == 0:
                continue
            Xu = self.X[i[rows]].toarray()
            yu = y[rows]
            # Same solution, cheaper system: with n_u < d ratings, the identity
            # (X^T X + lam I)^-1 X^T = X^T (X X^T + lam I)^-1 turns a d x d solve
            # into an n_u x n_u one. np.linalg.solve(A, b) solves A x = b without
            # forming the inverse, which is faster and numerically safer.
            if len(rows) < d:
                alpha = np.linalg.solve(Xu @ Xu.T + self.lam * np.eye(len(rows)), yu)
                self.theta[user] = Xu.T @ alpha
            else:
                self.theta[user] = np.linalg.solve(Xu.T @ Xu + self.lam * np.eye(d), Xu.T @ yu)
        return self

    def predict(self, u, i):
        """Predicted stars, clipped to 1-5, for each pair (u[k], i[k])."""
        u, i = np.asarray(u), np.asarray(i)
        # row-wise dot product x_i . theta_u for each (u, i) pair
        content = np.asarray(self.X[i].multiply(self.theta[u]).sum(axis=1)).ravel()
        return np.clip(self.bias.baseline(u, i) + content, MIN_STARS, MAX_STARS)

    def score_all(self, u):
        """Predicted stars for user u on every item (array of length n_items)."""
        base = self.bias.mu + self.bias.b_u[u] + self.bias.b_i
        return np.clip(base + self.X @ self.theta[u], MIN_STARS, MAX_STARS)
