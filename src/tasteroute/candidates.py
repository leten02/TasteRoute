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


def blend_popularity(pred, popularity, gamma):
    """Ranking score r_hat + gamma * log(1 + popularity).

    Rating-trained models predict how much a user would like a place, not how
    likely they are to go; popularity carries that second signal. gamma = 0
    keeps the pure rating order. np.log1p(x) computes log(1 + x) accurately.
    """
    return pred + gamma * np.log1p(popularity)


def two_stage(filt, u, pred, pool):
    """Take the `pool` most popular eligible places, then order them by pred.

    Returns the candidate ids, best first (same contract as CandidateFilter.top_m).
    """
    popular = filt.top_m(u, filt.popularity, m=pool)
    return popular[np.argsort(-pred[popular], kind="stable")]


def to_unit(scores):
    """Min-max rescale to [0, 1] so a blended score can feed rerank like a rating.

    rerank maps ratings with (r - 1) / 4; passing 1 + 4 * to_unit(s) makes the
    same formula see the blended score on the 1-5 scale. Equal scores map to 1,
    not 0: a 0 would zero every re-rank score and hide the distance order.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return s
    span = s.max() - s.min()
    return np.ones_like(s) if span == 0 else (s - s.min()) / span
