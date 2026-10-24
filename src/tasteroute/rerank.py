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
    """Weight in [0, 1] for distances d (km)."""
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
