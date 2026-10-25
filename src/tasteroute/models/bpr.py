"""Model-based CF for ranking: BPR matrix factorization on implicit feedback.

The rating models (bias, MF, CBF, hybrid) are trained to predict *how many
stars* a user would give, so they rank places a user might like rather than
places a user is likely to go to, and lose to plain popularity on P@10 (D1).
BPR learns the ranking directly from which places a user went to:

    x_ui = b_i + p_u . q_i        (score of item i for user u)

and, for a place i the user went to and a place j they did not, maximizes

    ln sigma(x_ui - x_uj) - lam * (||p_u||^2 + ||q_i||^2 + ||q_j||^2 + b_i^2 + b_j^2)

with stochastic gradient ascent over sampled (u, i, j) triples. The item bias
b_i lets the model learn popularity; p_u . q_i learns the personal part.

Reference: S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme,
"BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009.
"""
import numpy as np

from .visit import sigmoid


class BPR:
    """BPR-MF trained with mini-batch SGD and early stopping on a validation score.

    Args:
        n_users, n_items: sizes of the id spaces.
        f: number of latent factors.
        lr: learning rate.
        lam: L2 strength on factors and item biases.
        batch: (u, i, j) triples per gradient step.
        max_epochs: maximum passes over the positive pairs.
        patience: stop after this many epochs without a better validation score.
        seed: random seed (initialization, shuffling, negative sampling).
    """

    def __init__(self, n_users, n_items, f=32, lr=0.05, lam=1e-3, batch=1024, max_epochs=60, patience=3, seed=42):
        self.n_users, self.n_items = n_users, n_items
        self.f, self.lr, self.lam, self.batch = f, lr, lam, batch
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed

    def fit(self, train, val_score=None, verbose=False):
        """Fit on the (u, i) pairs in train (every row is a positive).

        val_score: optional function model -> number (higher is better), e.g.
        validation NDCG@10; enables early stopping and restores the best epoch.
        Records history (loss, val score per epoch) and best_epoch.
        """
        rng = np.random.default_rng(self.seed)
        u_all, i_all = train.u.values.astype(np.int64), train.i.values.astype(np.int64)
        # sorted keys u * n_items + i: "did u go to i?" by binary search (np.searchsorted)
        self._seen = np.unique(u_all * self.n_items + i_all)
        self.P = rng.normal(0, 0.1, (self.n_users, self.f))
        self.Q = rng.normal(0, 0.1, (self.n_items, self.f))
        self.b = np.zeros(self.n_items)
        self.history, self.best_epoch = [], None
        best, best_state, bad = -np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(len(u_all))
            losses = []
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                losses.append(self._step(u_all[idx], i_all[idx], rng))
            rec = {"epoch": epoch, "loss": float(np.mean(losses))}
            if val_score is not None:
                rec["val"] = float(val_score(self))
            self.history.append(rec)
            if verbose:
                print(rec, flush=True)
            if val_score is None:
                continue
            if rec["val"] > best:
                best, self.best_epoch, bad = rec["val"], epoch, 0
                best_state = (self.P.copy(), self.Q.copy(), self.b.copy())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.P, self.Q, self.b = best_state
        return self

    def _negatives(self, u, rng):
        """One item per user in u that the user has not gone to (uniform, resampled)."""
        j = rng.integers(self.n_items, size=len(u))
        for _ in range(10):
            keys = u * self.n_items + j
            pos = np.searchsorted(self._seen, keys)
            hit = (pos < len(self._seen)) & (self._seen[np.minimum(pos, len(self._seen) - 1)] == keys)
            if not hit.any():
                break
            j[hit] = rng.integers(self.n_items, size=int(hit.sum()))
        return j

    def _step(self, u, i, rng):
        """One mini-batch of gradient ascent; returns the mean -ln sigma(x_uij)."""
        j = self._negatives(u, rng)
        pu, qi, qj = self.P[u], self.Q[i], self.Q[j]
        x = self.b[i] - self.b[j] + np.sum(pu * (qi - qj), axis=1)
        # d/dx ln sigma(x) = sigma(-x): how much each triple still needs to move
        g = sigmoid(-x)[:, None]
        lr, lam = self.lr, self.lam
        # np.add.at(a, idx, v) adds v into a[idx], summing repeated indices (a user
        # or item can appear several times in one batch; a[idx] += v would not)
        np.add.at(self.P, u, lr * (g * (qi - qj) - lam * pu))
        np.add.at(self.Q, i, lr * (g * pu - lam * qi))
        np.add.at(self.Q, j, lr * (-g * pu - lam * qj))
        np.add.at(self.b, i, lr * (g[:, 0] - lam * self.b[i]))
        np.add.at(self.b, j, lr * (-g[:, 0] - lam * self.b[j]))
        return float(np.mean(np.logaddexp(0, -x)))  # -ln sigma(x) = log(1 + e^-x)

    def score_all(self, u):
        """Ranking score of every item for user u (higher = more likely to go).

        u = None (a new visitor without history) scores by the item bias alone.
        """
        return self.b.copy() if u is None or u < 0 else self.b + self.Q @ self.P[u]

    def save(self, path):
        """Write the fitted parameters to an .npz file."""
        np.savez_compressed(path, P=self.P, Q=self.Q, b=self.b, f=self.f, lam=self.lam)

    @classmethod
    def load(cls, path):
        """Rebuild a fitted model from save()."""
        z = np.load(path)
        m = cls(z["P"].shape[0], z["Q"].shape[0], f=int(z["f"]), lam=float(z["lam"]))
        m.P, m.Q, m.b = z["P"], z["Q"], z["b"]
        return m
