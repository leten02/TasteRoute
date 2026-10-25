"""Distance-aware re-ranking of the candidate list.

score = (r_hat - 1) / 4 * w(d): the predicted rating rescaled to [0, 1], so a
weight of 0.5 always halves the score, times a decreasing distance weight.

Decay shapes, each with a distance scale tau (km):
    exp       w = exp(-d / tau)              (default; halves every tau * ln 2)
    gaussian  w = exp(-d^2 / (2 tau^2))
    power     w = 1 / (1 + d / tau)
    cutoff    w = 1 if d <= tau else 0
Proposal 6.3.
"""
import numpy as np

DECAYS = ("exp", "gaussian", "power", "cutoff")


def distance_weight(d, tau, decay="exp"):
    """Weight in [0, 1] for distances d (km); tau must be positive."""
    if tau <= 0:
        raise ValueError("tau must be positive (tau = 0 divides by zero, tau < 0 rewards distance)")
    d = np.asarray(d, dtype=float)
    if decay == "exp":
        return np.exp(-d / tau)
    if decay == "gaussian":
        return np.exp(-(d ** 2) / (2 * tau ** 2))
    if decay == "power":
        return 1.0 / (1.0 + d / tau)
    if decay == "cutoff":
        return (d <= tau).astype(float)
    raise ValueError(f"decay must be one of {DECAYS}")


def rerank(candidates, pred, dist, tau, decay="exp", k=10):
    """Re-order candidates by rescaled rating x distance weight; return the top k.

    Args:
        candidates: item ids from candidates.CandidateFilter.top_m.
        pred: predicted ratings of those candidates (same order).
        dist: distance (km) from the user's anchor to each candidate.
    """
    score = (np.asarray(pred) - 1.0) / 4.0 * distance_weight(dist, tau, decay)
    order = np.argsort(-score, kind="stable")[:k]
    return np.asarray(candidates)[order]


def mmr(candidates, relevance, features, lam, k=10):
    """Maximal Marginal Relevance: trade relevance for variety, one pick at a time.

    Each step picks the candidate maximizing
        (1 - lam) * relevance_i - lam * max_{j already picked} sim(i, j)
    with sim the cosine similarity of the feature rows (categories and
    attributes). lam = 0 is plain relevance order; larger lam avoids picking
    six near-identical places. relevance should be on a 0-1 scale.

    Reference: J. Carbonell, J. Goldstein, "The Use of MMR, Diversity-Based
    Reranking for Reordering Documents and Producing Summaries", SIGIR 1998.

    Args:
        candidates: item ids.
        relevance: score per candidate (same order), 0-1.
        features: item x feature matrix (scipy sparse or dense), rows by item id.
    """
    cand = np.asarray(candidates)
    rel = np.asarray(relevance, dtype=float)
    k = min(k, len(cand))
    if k == 0:
        return cand[:0]
    F = features[cand]
    F = F.toarray() if hasattr(F, "toarray") else np.asarray(F, dtype=float)
    # np.linalg.norm(F, axis=1): the length of each row (one per candidate), used
    # to scale rows to unit length so F @ F.T is the cosine similarity
    norms = np.linalg.norm(F, axis=1)
    norms[norms == 0] = 1.0
    F = F / norms[:, None]
    sim = F @ F.T
    picked = [int(np.argmax(rel))]
    max_sim = sim[picked[0]].copy()
    for _ in range(1, k):
        gain = (1 - lam) * rel - lam * max_sim
        gain[picked] = -np.inf
        nxt = int(np.argmax(gain))
        picked.append(nxt)
        max_sim = np.maximum(max_sim, sim[nxt])
    return cand[picked]
