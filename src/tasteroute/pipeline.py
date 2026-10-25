"""End-to-end day plan for one person: scores -> candidates -> distance re-rank -> route.

This is the configuration the experiments found best (experiments/popularity.py,
experiments/ranking.py, experiments/route.py):
    1. blend predicted ratings with log popularity (gamma = 5)
    2. take the top 50 eligible places, re-rank by exp distance decay (tau = 2 km)
    3. keep the top n and order them with Held-Karp from the start point,
       on walking distances along streets when a road network is given
       (roads.py) or on straight-line distances otherwise
"""
import numpy as np

from .candidates import blend_popularity, to_unit
from .geo import distance_matrix, haversine
from .rerank import rerank
from .roads import RoadNetwork
from .route import held_karp, path_length

GAMMA, TAU, DECAY = 5.0, 2.0, "exp"
MAX_PLACES = 12  # Held-Karp is O(n^2 2^n): 12 places take well under a second


def plan_day(pred, filt, items, start, user=None, n=6, mask=None, gamma=GAMMA, tau=TAU, decay=DECAY, roads=None):
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
            network around the chosen places, or a roads.RoadNetwork.

    Returns:
        dict with the places in visiting order (ids), the route length and the
        length if the same places were visited in recommendation order (both
        walking km when roads is set), the straight-line route length, and
        `path`, the (lat, lon) points to draw (along streets when roads is set).

    Re-ranking (step 2) still uses straight-line distance: it only has to tell
    near from far among 50 candidates, and street distance for all of them
    would need a much larger network.
    """
    if not 1 <= n <= MAX_PLACES:
        raise ValueError(f"n must be between 1 and {MAX_PLACES}, got {n}")
    score = blend_popularity(np.asarray(pred, dtype=float), filt.popularity, gamma)
    if mask is not None:
        score = np.where(mask, score, -np.inf)
    cand = filt.top_m(-1 if user is None else user, score)
    lat, lon = items.lat.values, items.lon.values
    dist = haversine(start[0], start[1], lat[cand], lon[cand])
    chosen = rerank(cand, 1 + 4 * to_unit(score[cand]), dist, tau, decay, k=n)

    # np.r_[a, b] concatenates a scalar and an array into one 1-D array: the start
    # point becomes node 0, followed by the n chosen places
    pts_lat, pts_lon = np.r_[start[0], lat[chosen]], np.r_[start[1], lon[chosen]]
    straight = distance_matrix(pts_lat, pts_lon)
    routes = None
    if roads is not None and len(chosen) > 0:  # nothing to route: no street download
        net = RoadNetwork.around(pts_lat, pts_lon) if isinstance(roads, str) else roads
        routes = net.routes(pts_lat, pts_lon)
    D = straight if routes is None else routes.D

    order, length = held_karp(D)
    path = [(pts_lat[k], pts_lon[k]) for k in [0, *order]] if routes is None else routes.tour_path(order)
    return {
        "places": [int(chosen[k - 1]) for k in order],
        "route_km": length,
        "rec_order_km": path_length(D, range(1, len(chosen) + 1)),
        "straight_km": path_length(straight, order),
        "path": path,
    }
