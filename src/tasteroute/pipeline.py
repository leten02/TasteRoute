"""End-to-end day plan for one person: scores -> candidates -> distance re-rank -> route.

This is the configuration the experiments found best (experiments/popularity.py,
experiments/ranking.py, experiments/route.py):
    1. blend predicted ratings with log popularity (gamma = 5)
    2. take the top 200 eligible places and re-rank them by exp distance decay
       (tau = 2 km), or, with a learned visit model (models/visit.py), by its
       logit over distance, travel radius, predicted rating and popularity.
       200 rather than 50: with only 50 citywide candidates there may be no
       near place left to promote (experiments/learned_distance.py: NDCG@10
       0.0340 -> 0.0373 for the fixed rule, tau re-tuned on validation)
    3. keep the top n and order them with Held-Karp from the start point,
       on walking distances along streets when a road network is given
       (roads.py) or on straight-line distances otherwise
"""
import numpy as np

from .candidates import blend_popularity, to_unit
from .geo import distance_matrix, haversine
from .rerank import rerank
from .models.visit import rank_learned
from .roads import RoadNetwork
from .route import held_karp, path_length

GAMMA, TAU, DECAY = 5.0, 2.0, "exp"
MAX_PLACES = 12  # Held-Karp is O(n^2 2^n): 12 places take well under a second
POOL = 200


def plan_day(pred, filt, items, start, user=None, n=6, mask=None, gamma=GAMMA, tau=TAU, decay=DECAY, roads=None,
             visit=None):
    """Recommend n places near `start` and the shortest order to visit them.

    Args:
        pred: predicted rating for every item (model.score_all(user)).
        filt: candidates.CandidateFilter (open, enough ratings, not yet visited).
        items: DataFrame from data.load_items().
        start: (lat, lon) where the day begins (user anchor, hotel, ...).
        user: user id whose visited places are excluded; None for a new visitor.
        n: number of places to visit.
        mask: optional boolean array over items, e.g. only bakeries.
        roads: None for straight-line routing, "osm" to download the walking
            network around the chosen places, a roads.RoadNetwork, or a
            function (lats, lons) -> RoadNetwork (e.g. to limit download time).
        visit: None for the fixed distance rule, or (model, features, columns)
            to rank with a learned visit model: a fitted
            models.visit.LogisticRegression, a models.visit.VisitFeatures and
            the feature columns it was trained on.

    Returns:
        dict with
            places        ids in visiting order
            ranked        the same ids in recommendation order
            route_km      length of the visiting order (walking km when roads is set)
            rec_order_km  length if visited in recommendation order
            straight_km   straight-line length of the visiting order
            legs_km       length of each leg: start -> 1st, 1st -> 2nd, ...
            path          (lat, lon) points to draw, along streets when roads is set
            rec_path      the same for the recommendation order

    Re-ranking (step 2) still uses straight-line distance: it only has to tell
    near from far among the candidates, and street distance for all of them
    would need a much larger network.
    """
    if not 1 <= n <= MAX_PLACES:
        raise ValueError(f"n must be between 1 and {MAX_PLACES}, got {n}")
    score = blend_popularity(np.asarray(pred, dtype=float), filt.popularity, gamma)
    if mask is not None:
        score = np.where(mask, score, -np.inf)
    lat, lon = items.lat.values, items.lon.values
    u = -1 if user is None else user
    cand = filt.top_m(u, score, m=POOL)
    dist = haversine(start[0], start[1], lat[cand], lon[cand])
    if visit is None:
        chosen = rerank(cand, 1 + 4 * to_unit(score[cand]), dist, tau, decay, k=n)
    else:
        model, feats, columns = visit
        X = feats.build(user, cand, np.asarray(pred, dtype=float)[cand], dist=dist)
        chosen = rank_learned(model, X, cand, k=n, columns=columns)

    # np.r_[a, b] concatenates a scalar and an array into one 1-D array: the start
    # point becomes node 0, followed by the n chosen places
    pts_lat, pts_lon = np.r_[start[0], lat[chosen]], np.r_[start[1], lon[chosen]]
    straight = distance_matrix(pts_lat, pts_lon)
    routes = None
    if roads is not None:
        if isinstance(roads, str):
            net = RoadNetwork.around(pts_lat, pts_lon)
        elif callable(roads):
            net = roads(pts_lat, pts_lon)
        else:
            net = roads
        routes = net.routes(pts_lat, pts_lon)
    D = straight if routes is None else routes.D

    order, length = held_karp(D)
    rec = list(range(1, len(chosen) + 1))

    def draw(o):
        return [(pts_lat[k], pts_lon[k]) for k in [0, *o]] if routes is None else routes.tour_path(o)

    stops = [0, *order]
    return {
        "places": [int(chosen[k - 1]) for k in order],
        "ranked": [int(c) for c in chosen],
        "route_km": length,
        "rec_order_km": path_length(D, rec),
        "straight_km": path_length(straight, order),
        "legs_km": [float(D[a, b]) for a, b in zip(stops[:-1], stops[1:])],
        "path": draw(order),
        "rec_path": draw(rec),
    }
