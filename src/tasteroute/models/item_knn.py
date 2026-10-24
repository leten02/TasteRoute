"""Memory-based CF: item-based k-nearest neighbors on top of the bias baseline.

    r_hat_ui = b_ui + sum_{j in N_k(i;u)} s_ij (r_uj - b_uj) / sum_{j in N_k(i;u)} |s_ij|

N_k(i;u) is the k items u rated that are most similar to i. Similarities are
shrunk toward 0 when few users rated both items:  s'_ij = n_ij / (n_ij + beta) * s_ij.

Three similarities differ only in how ratings are centered before the cosine:
    cosine    raw stars
    adjusted  stars minus the user's mean
    pearson   stars minus the item's mean

The cosine is normalized by each item's full rating vector (standard cosine
over the item-user matrix), not only by the co-rated users of each pair. Both
are common; the full-vector version is one sparse matrix product.

Proposal section 5.2. Reference: Sarwar et al., "Item-based collaborative
filtering recommendation algorithms", WWW 2001; Koren, "Factor in the
neighbors", TKDD 2010 (shrinkage and the bias baseline).
"""
import numpy as np
from scipy import sparse

from .baseline import MAX_STARS, MIN_STARS, BiasModel

SIMILARITIES = ("cosine", "adjusted", "pearson")


class ItemKNN:
    """Item-based kNN.

    Args:
        n_users, n_items: array sizes (see data.n_users_items).
        similarity: "cosine", "adjusted" or "pearson".
        k: number of neighbors used per prediction. Can be changed after fit.
        beta: shrinkage strength (0 = no shrinkage).
        bias_lam: lambda of the bias model that provides b_ui.
    """

    def __init__(self, n_users, n_items, similarity="cosine", k=40, beta=25.0, bias_lam=5.0):
        if similarity not in SIMILARITIES:
            raise ValueError(f"similarity must be one of {SIMILARITIES}")
        self.n_users, self.n_items = n_users, n_items
        self.similarity, self.k, self.beta, self.bias_lam = similarity, k, beta, bias_lam

    def fit(self, train):
        u, i = train.u.values, train.i.values
        r = train.stars.values.astype(float)
        self.bias = BiasModel(self.n_users, self.n_items, lam=self.bias_lam).fit(train)

        x = r.copy()
        if self.similarity == "adjusted":
            x -= _group_mean(u, r, self.n_users)[u]
        elif self.similarity == "pearson":
            x -= _group_mean(i, r, self.n_items)[i]

        # sparse.csr_matrix((values, (rows, cols)), shape) builds a sparse matrix
        # from coordinate triples; only the listed cells are stored.
        X = sparse.csr_matrix((x, (i, u)), shape=(self.n_items, self.n_users))
        B = sparse.csr_matrix((np.ones_like(r), (i, u)), shape=(self.n_items, self.n_users))

        norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norm[norm == 0] = 1.0
        # dense 7,693 x 7,693 float32 arrays are about 237 MB each
        S = (X @ X.T).toarray().astype(np.float32)
        S /= norm[:, None].astype(np.float32)
        S /= norm[None, :].astype(np.float32)
        if self.beta > 0:
            N = (B @ B.T).toarray().astype(np.float32)  # n_ij: users who rated both
            S *= N / (N + np.float32(self.beta))
            del N
        np.fill_diagonal(S, 0.0)
        self.S = S

        # per-user rated items and their residuals r_uj - b_uj
        order = np.argsort(u, kind="stable")
        res = r - self.bias.baseline(u, i)
        starts = np.searchsorted(u[order], np.arange(self.n_users + 1))
        # Sort once: each slice below is a view that keeps its base array alive,
        # so re-sorting per user would pin one full copy per user in memory.
        i_sorted, res_sorted = i[order], res[order]
        self._items = [i_sorted[starts[x]:starts[x + 1]] for x in range(self.n_users)]
        self._res = [res_sorted[starts[x]:starts[x + 1]] for x in range(self.n_users)]
        return self

    def _predict_user(self, u, items):
        """Predictions for one user and an array of items."""
        base = self.bias.baseline(u, items)
        J, res = self._items[u], self._res[u]
        if len(J) == 0:
            return base
        sims = self.S[np.ix_(items, J)]  # np.ix_ selects the rows x cols sub-block
        if len(J) > self.k:
            # np.argpartition(a, kth) puts the kth smallest in place without a full
            # sort; on -sims it picks the k largest similarities per row.
            top = np.argpartition(-sims, self.k - 1, axis=1)[:, : self.k]
            sims = np.take_along_axis(sims, top, axis=1)
            res = res[top]
        else:
            res = np.broadcast_to(res, sims.shape)
        den = np.abs(sims).sum(axis=1)
        num = (sims * res).sum(axis=1)
        adj = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
        return base + adj  # no neighbor (den == 0) falls back to b_ui

    def predict(self, u, i):
        u, i = np.asarray(u), np.asarray(i)
        out = np.empty(len(u))
        order = np.argsort(u, kind="stable")
        bounds = np.flatnonzero(np.diff(u[order])) + 1
        for idx in np.split(order, bounds):
            out[idx] = self._predict_user(u[idx[0]], i[idx])
        return np.clip(out, MIN_STARS, MAX_STARS)

    def score_all(self, u):
        return np.clip(self._predict_user(u, np.arange(self.n_items)), MIN_STARS, MAX_STARS)


def _group_mean(ids, values, n):
    cnt = np.bincount(ids, minlength=n)
    tot = np.bincount(ids, weights=values, minlength=n)
    return np.divide(tot, cnt, out=np.zeros(n), where=cnt > 0)
