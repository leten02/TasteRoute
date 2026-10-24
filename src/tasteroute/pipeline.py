"""End-to-end day plan for one person: scores -> candidates -> distance re-rank -> route.

This is the configuration the experiments found best (experiments/popularity.py,
experiments/ranking.py, experiments/route.py):
    1. blend predicted ratings with log popularity (gamma = 5)
    2. take the top 50 eligible places, re-rank by exp distance decay (tau = 2 km)
    3. keep the top n and order them with Held-Karp from the start point
"""
import numpy as np

from .candidates import blend_popularity, to_unit
from .geo import distance_matrix, haversine
from .rerank import rerank
from .route import held_karp, path_length

GAMMA, TAU, DECAY = 5.0, 2.0, "exp"


def plan_day(pred, filt, items, start, user=None, n=6, mask=None, gamma=GAMMA, tau=TAU, decay=DECAY):
    """Recommend n places near `start` and the shortest order to visit them.

    Args:
        pred: predicted rating for every item (model.score_all(user)).
        filt: candidates.CandidateFilter (open, enough ratings, not yet visited).
        items: DataFrame from data.load_items().
        start: (lat, lon) where the day begins (user anchor, hotel, ...).
        user: user id whose visited places are excluded; None for a new visitor.
        n: number of places to visit.
        mask: optional boolean array over items, e.g. only bakeries.

    Returns:
        dict with the places in visiting order (ids), the route length, and the
        length if the same places were visited in recommendation order.
    """
    score = blend_popularity(np.asarray(pred, dtype=float), filt.popularity, gamma)
    if mask is not None:
        score = np.where(mask, score, -np.inf)
    cand = filt.top_m(-1 if user is None else user, score)
    lat, lon = items.lat.values, items.lon.values
    dist = haversine(start[0], start[1], lat[cand], lon[cand])
    chosen = rerank(cand, 1 + 4 * to_unit(score[cand]), dist, tau, decay, k=n)

    # np.r_[a, b] concatenates a scalar and an array into one 1-D array: the start
    # point becomes node 0, followed by the n chosen places
    D = distance_matrix(np.r_[start[0], lat[chosen]], np.r_[start[1], lon[chosen]])
    order, length = held_karp(D)
    return {
        "places": [int(chosen[k - 1]) for k in order],
        "route_km": length,
        "rec_order_km": path_length(D, range(1, len(chosen) + 1)),
    }
