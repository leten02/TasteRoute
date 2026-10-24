"""Hybrid: weighted blend of a CF model and the CBF model.

Fixed weight:     r_hat = alpha * r_hat_CF + (1 - alpha) * r_hat_CBF
Adaptive weight:  alpha(n_u) = n_u / (n_u + kappa)

The adaptive form trusts CF more as a user rates more places; kappa is the
number of ratings at which CF and CBF get equal weight. Example: n_u = 45 and
kappa = 100 give alpha = 45 / 145 = 0.31.

Both component models are fitted beforehand; the hybrid only blends them.
Proposal section 5.5.
"""
import numpy as np

from .baseline import MAX_STARS, MIN_STARS


class Hybrid:
    """Blend two fitted recommenders.

    Args:
        cf, cbf: fitted models with predict(u, i) and score_all(u).
        alpha: fixed CF weight in [0, 1]; used when kappa is None.
        kappa: if set, use the adaptive weight n_u / (n_u + kappa) instead.
    """

    def __init__(self, cf, cbf, alpha=0.5, kappa=None):
        if kappa is not None and kappa <= 0:
            raise ValueError("kappa must be positive (kappa = 0 gives 0/0 for users without ratings)")
        self.cf, self.cbf = cf, cbf
        self.alpha, self.kappa = alpha, kappa

    def fit(self, train):
        """Store each user's number of training ratings n_u (no model is retrained)."""
        self.n_u = np.bincount(train.u.values, minlength=self.cf.n_users)
        self.n_items = self.cf.n_items
        return self

    def weight(self, u):
        """CF weight alpha for each user in u."""
        if self.kappa is None:
            return np.full(np.shape(u), float(self.alpha))
        n = self.n_u[u]
        return n / (n + self.kappa)

    def predict(self, u, i):
        u = np.asarray(u)
        a = self.weight(u)
        blend = a * self.cf.predict(u, i) + (1 - a) * self.cbf.predict(u, i)
        return np.clip(blend, MIN_STARS, MAX_STARS)

    def score_all(self, u):
        a = float(self.weight(np.array([u]))[0])
        return a * self.cf.score_all(u) + (1 - a) * self.cbf.score_all(u)
