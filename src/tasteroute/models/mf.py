"""Model-based CF: biased matrix factorization trained with SGD.

    r_hat = mu + b_u + b_i + p_u . q_i

Loss: sum (r - r_hat)^2 + lam * (b_u^2 + b_i^2 + |p_u|^2 + |q_i|^2).
For one observed rating with error e = r - r_hat and learning rate eta:

    b_u += eta * (e - lam * b_u)          p_u += eta * (e * q_i - lam * p_u)
    b_i += eta * (e - lam * b_i)          q_i += eta * (e * p_u - lam * q_i)

Proposal section 5.3. Reference for the update rule: Koren, Bell, Volinsky,
"Matrix Factorization Techniques for Recommender Systems", IEEE Computer 2009,
and https://albertauyeung.github.io/2017/04/23/python-matrix-factorization.html/
"""
import numpy as np

from ..metrics import rmse
from .baseline import MAX_STARS, MIN_STARS


class MF:
    """Biased MF with per-rating SGD and early stopping on validation RMSE.

    Args:
        n_users, n_items: array sizes (see data.n_users_items).
        f: number of latent factors.
        eta: learning rate.
        lam: L2 regularization strength.
        max_epochs: upper bound on passes over the training ratings.
        patience: stop after this many epochs without a validation improvement,
            then restore the parameters from the best epoch.
        seed: seed for the initial factors and the per-epoch shuffle.
    """

    def __init__(self, n_users, n_items, f=10, eta=0.01, lam=0.1, max_epochs=50, patience=3, seed=42):
        self.n_users, self.n_items = n_users, n_items
        self.f, self.eta, self.lam = f, eta, lam
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed

    def fit(self, train, val=None, verbose=False):
        """Train on train; if val is given, early-stop on its RMSE.

        Records self.history: one dict per epoch with train and val RMSE (figure M3).
        """
        # np.random.default_rng(seed) is numpy's seeded random generator;
        # rng.normal(mean, std, shape) draws Gaussian values, rng.permutation(n)
        # returns 0..n-1 in random order.
        rng = np.random.default_rng(self.seed)
        self.mu = float(train.stars.mean())
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.P = rng.normal(0, 0.1, (self.n_users, self.f))
        self.Q = rng.normal(0, 0.1, (self.n_items, self.f))

        u_all, i_all = train.u.values, train.i.values
        r_all = train.stars.values.astype(float)
        self.history, self.best_epoch = [], None
        best, best_state, bad = np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(len(r_all))
            self._sgd_epoch(u_all[order], i_all[order], r_all[order])

            rec = {"epoch": epoch, "train_rmse": rmse(r_all, self.predict(u_all, i_all))}
            if val is not None:
                rec["val_rmse"] = rmse(val.stars.values, self.predict(val.u.values, val.i.values))
            self.history.append(rec)
            if verbose:
                print(rec)

            if val is None:
                continue
            # keep a checkpoint at every new minimum, so the restored parameters
            # always match the lowest val_rmse in history; patience only counts
            # epochs that improve by less than 1e-5
            improved = rec["val_rmse"] < best - 1e-5
            if rec["val_rmse"] < best:
                best, self.best_epoch = rec["val_rmse"], epoch
                best_state = (self.b_u.copy(), self.b_i.copy(), self.P.copy(), self.Q.copy())
            bad = 0 if improved else bad + 1
            if bad >= self.patience:
                break
        if best_state is not None:
            self.b_u, self.b_i, self.P, self.Q = best_state
        return self

    def _sgd_epoch(self, us, is_, rs):
        """One pass of per-rating SGD updates (plain Python loop, in place)."""
        eta, lam, mu = self.eta, self.lam, self.mu
        b_u, b_i, P, Q = self.b_u, self.b_i, self.P, self.Q
        for u, i, r in zip(us.tolist(), is_.tolist(), rs.tolist()):
            pu, qi = P[u], Q[i]
            e = r - (mu + b_u[u] + b_i[i] + pu @ qi)
            b_u[u] += eta * (e - lam * b_u[u])
            b_i[i] += eta * (e - lam * b_i[i])
            # both factor updates must use the old p_u, so compute them first
            dp = eta * (e * qi - lam * pu)
            Q[i] = qi + eta * (e * pu - lam * qi)
            P[u] = pu + dp

    def predict(self, u, i):
        """Predicted stars, clipped to 1-5, for each pair (u[k], i[k])."""
        # sum over the last axis: works for one (u, i) pair and for arrays of pairs
        raw = self.mu + self.b_u[u] + self.b_i[i] + np.sum(self.P[u] * self.Q[i], axis=-1)
        return np.clip(raw, MIN_STARS, MAX_STARS)

    def score_all(self, u):
        """Predicted stars for user u on every item (array of length n_items)."""
        return np.clip(self.mu + self.b_u[u] + self.b_i + self.Q @ self.P[u], MIN_STARS, MAX_STARS)
