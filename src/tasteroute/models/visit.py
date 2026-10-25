"""Learned distance-aware ranking: will this user visit (and like) this place?

The fixed rule in rerank.py multiplies the predicted rating by exp(-d / tau)
with tau picked by hand. Here the trade-off between distance, predicted rating
and popularity is learned with logistic regression instead:

    P(visit | u, i) = sigmoid(w . x(u, i) + b)

Features x(u, i) (VisitFeatures.build):
    log1p(d)          distance (km) from the user's anchor, on a log scale so
                      the first kilometre matters more than the tenth
    d                 the same distance, linear: with log1p(d) the model can
                      learn both a sharp near-by drop and a steady far-away
                      penalty (the fixed rule's exp(-d / tau) is linear in d
                      on the log scale)
    log1p(d / rho_u)  the same distance relative to the user's own travel
                      radius rho_u (median distance of their training places
                      from their anchor): a user who usually travels 5 km
                      minds 3 km less than one who stays within 1 km
    r_hat             predicted rating from the rating model (Hybrid)
    log1p(pop)        number of training ratings of the place

The logistic regression is written out here (batch gradient descent on the
L2-regularized cross-entropy loss) rather than taken from a library, as in the
course.
"""
import numpy as np

from ..geo import haversine

FEATURES = ("log_dist", "dist", "log_dist_rel", "r_hat", "log_pop")
MIN_RADIUS_KM = 0.5


def sigmoid(z):
    """1 / (1 + e^-z), computed without overflow for large |z|."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    """Binary logistic regression trained by batch gradient descent.

    Loss (mean cross-entropy plus L2 on the weights, not the bias):
        L = -mean(y log p + (1 - y) log(1 - p)) + lam / 2 * ||w||^2
    Gradient: dL/dw = X^T (p - y) / n + lam * w,  dL/db = mean(p - y).
    Features are standardized (zero mean, unit variance) before training so
    one learning rate suits all of them; coef() maps the weights back.

    Step size: backtracking line search. A step must lower the loss by at least
    lr / 2 * ||grad||^2 (Armijo condition), otherwise lr is halved and the step
    retried; after an accepted step lr is doubled. Visits are rare (about 1 in
    200 candidates), which makes the loss very flat, and a fixed small step
    would need tens of thousands of iterations.

    Args:
        lam: L2 strength.
        lr: initial gradient descent step size.
        max_iter: maximum number of accepted steps.
        tol: stop when the gradient norm falls below this.
    """

    def __init__(self, lam=1e-3, lr=1.0, max_iter=2000, tol=1e-6):
        self.lam, self.lr, self.max_iter, self.tol = lam, lr, max_iter, tol

    def fit(self, X, y):
        """Fit on features X (n x d) and 0/1 labels y; records the loss history."""
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Z = (X - self.mean) / self.std
        n, d = Z.shape
        self.w, self.b = np.zeros(d), np.log((y.mean() + 1e-12) / (1 - y.mean() + 1e-12))
        self.history = []
        lr = self.lr
        loss = self._loss(Z, y, self.w, self.b)
        for _ in range(self.max_iter):
            self.history.append(loss)
            err = sigmoid(Z @ self.w + self.b) - y
            gw, gb = Z.T @ err / n + self.lam * self.w, err.mean()
            g2 = gw @ gw + gb * gb
            if np.sqrt(g2) < self.tol:
                break
            while True:  # backtracking line search
                w_new, b_new = self.w - lr * gw, self.b - lr * gb
                new_loss = self._loss(Z, y, w_new, b_new)
                if new_loss <= loss - lr / 2 * g2 or lr < 1e-12:
                    break
                lr /= 2
            self.w, self.b, loss = w_new, b_new, new_loss
            lr *= 2
        self.history.append(loss)
        return self

    def _loss(self, Z, y, w, b):
        """Mean cross-entropy plus L2, written with logaddexp so it never overflows.

        np.logaddexp(0, z) = log(1 + e^z); -log p = log(1 + e^-z), -log(1-p) = log(1 + e^z).
        """
        z = Z @ w + b
        ce = np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))
        return float(ce + self.lam / 2 * w @ w)

    def decision_function(self, X):
        """Logit w . x + b (higher = more likely); enough for ranking."""
        return ((np.asarray(X, dtype=float) - self.mean) / self.std) @ self.w + self.b

    def predict_proba(self, X):
        """P(y = 1 | x)."""
        return sigmoid(self.decision_function(X))

    def coef(self):
        """Weights and bias in the original (unstandardized) feature units."""
        w = self.w / self.std
        return w, float(self.b - w @ self.mean)

    def to_dict(self):
        """Plain-JSON parameters, enough to rebuild the model with from_dict."""
        return {"lam": self.lam, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "w": self.w.tolist(), "b": float(self.b)}

    @classmethod
    def from_dict(cls, d):
        """Rebuild a fitted model from to_dict() output."""
        m = cls(lam=d["lam"])
        m.mean, m.std = np.array(d["mean"]), np.array(d["std"])
        m.w, m.b = np.array(d["w"]), float(d["b"])
        return m


class VisitFeatures:
    """Builds x(u, i) for any user and candidate places.

    Args:
        items: DataFrame from data.load_items().
        train: training ratings (anchors, travel radii).
        anchors: (n_users, 2) array from geo.user_anchors.
        popularity: training ratings per item (CandidateFilter.popularity).
    """

    def __init__(self, items, train, anchors, popularity):
        n_items = int(items.i.max()) + 1
        self.lat = np.full(n_items, np.nan)
        self.lon = np.full(n_items, np.nan)
        self.lat[items.i.values], self.lon[items.i.values] = items.lat.values, items.lon.values
        self.anchors = anchors
        self.log_pop = np.log1p(popularity)
        d = haversine(anchors[train.u.values, 0], anchors[train.u.values, 1],
                      self.lat[train.i.values], self.lon[train.i.values])
        radius = np.full(len(anchors), np.nan)
        med = train.assign(d=d).groupby("u").d.median()
        radius[med.index.values] = med.values
        # users with one training place (radius 0) or none get the floor / the median user
        self.default_radius = float(np.nanmedian(radius))
        self.radius = np.maximum(np.where(np.isnan(radius), self.default_radius, radius), MIN_RADIUS_KM)

    def distance(self, u, cand):
        """Straight-line km from user u's anchor to each candidate."""
        return haversine(self.anchors[u, 0], self.anchors[u, 1], self.lat[cand], self.lon[cand])

    def build(self, u, cand, r_hat, dist=None, radius=None):
        """Feature matrix (len(cand) x 5) in FEATURES order.

        Args:
            u: user id (anchor and radius), or None with dist and radius given.
            cand: candidate item ids.
            r_hat: predicted ratings of the candidates (same order).
            dist: optional distances (km), e.g. from a custom start point.
            radius: optional travel radius (km); defaults to the user's own.
        """
        d = self.distance(u, cand) if dist is None else np.asarray(dist, dtype=float)
        rho = (self.radius[u] if u is not None else self.default_radius) if radius is None else radius
        return np.column_stack([np.log1p(d), d, np.log1p(d / rho), np.asarray(r_hat, dtype=float),
                                self.log_pop[cand]])


def rank_learned(model, X, candidates, k=10, columns=None):
    """Top k candidates by the learned visit logit.

    columns: optional indices of the features the model was trained on
    (for the ablation without the personal radius).
    """
    Xc = X if columns is None else X[:, columns]
    order = np.argsort(-model.decision_function(Xc), kind="stable")[:k]
    return np.asarray(candidates)[order]
