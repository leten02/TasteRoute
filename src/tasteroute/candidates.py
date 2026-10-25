"""Candidate generation: the top M places per user before re-ranking.

Only places a user can still be recommended are kept: open, not already rated
by the user in train or val, and with at least MIN_SUPPORT training ratings
(rating models over-score places that few users rated). Proposal 5.6, 9.1.
"""
import numpy as np

M_CANDIDATES = 50
MIN_SUPPORT = 20


class CandidateFilter:
    """Precomputed eligibility mask plus each user's already-seen items.

    Args:
        items: DataFrame from data.load_items().
        train: training ratings (support counts, seen items).
        seen: ratings whose items must not be recommended again (train + val).
        min_support: minimum number of training ratings per place.
    """

    def __init__(self, items, train, seen, min_support=MIN_SUPPORT):
        n_items = int(items.i.max()) + 1
        support = np.bincount(train.i.values, minlength=n_items)
        is_open = np.zeros(n_items, dtype=bool)
        is_open[items.i.values] = items.is_open.values == 1
        self.eligible = is_open & (support >= min_support)
        self.popularity = support.astype(float)
        self.seen = seen.groupby("u").i.apply(np.asarray).to_dict()

    def top_m(self, u, scores, m=M_CANDIDATES):
        """Indices of the m highest-scoring eligible, unseen places, best first."""
        s = np.where(self.eligible, scores, -np.inf)
        s[self.seen.get(u, [])] = -np.inf
        top = np.argpartition(-s, m - 1)[:m]
        top = top[np.argsort(-s[top], kind="stable")]
        return top[np.isfinite(s[top])]
