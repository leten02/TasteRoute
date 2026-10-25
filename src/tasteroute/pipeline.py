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
    3. keep the top n (optionally trading relevance for variety with MMR) and
       order them with Held-Karp from the start point, on walking distances
       along streets when a road network is given (roads.py) or on
       straight-line distances otherwise
For several days, plan_days groups the recommended places into neighborhoods
with k-means (cluster.py) and routes each group as one day.
"""
import numpy as np

from .candidates import blend_popularity, to_unit
from .geo import distance_matrix, haversine
from .cluster import balanced_kmeans, to_km
from .models.visit import rank_learned
from .rerank import mmr, rerank_scores
from .roads import RoadNetwork
from .route import held_karp, path_length

GAMMA, TAU, DECAY = 5.0, 2.0, "exp"
MAX_PLACES = 12  # Held-Karp is O(n^2 2^n): 12 places take well under a second
POOL = 200


def recommend(pred, filt, items, start, user=None, n=6, mask=None, gamma=GAMMA, tau=TAU, decay=DECAY,
              visit=None, diversity=0.0, features=None):
    """Steps 1-2 plus the top-n cut: the n places to visit, best first.

    Args are those of plan_day; additionally
        diversity: MMR weight in [0, 1); 0 keeps the plain relevance order.
        features: item x feature matrix for MMR similarity (needed if diversity > 0).
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    score = blend_popularity(np.asarray(pred, dtype=float), filt.popularity, gamma)
    if mask is not None:
        score = np.where(mask, score, -np.inf)
    lat, lon = items.lat.values, items.lon.values
    cand = filt.top_m(-1 if user is None else user, score, m=POOL)
    dist = haversine(start[0], start[1], lat[cand], lon[cand])
    if visit is None:
        relevance = rerank_scores(1 + 4 * to_unit(score[cand]), dist, tau, decay)  # as experiments' rerank()
    else:
        model, feats, columns = visit
        X = feats.build(user, cand, np.asarray(pred, dtype=float)[cand], dist=dist)
        if diversity <= 0:
            return rank_learned(model, X, cand, k=n, columns=columns)
        relevance = model.decision_function(X if columns is None else X[:, columns])
    if diversity > 0:
        if features is None:
            raise ValueError("diversity > 0 needs the item features for similarity")
        # MMR weighs relevance against similarity (0-1), so put relevance on 0-1
        # too; otherwise the same slider value would mean more at small tau
        return mmr(cand, to_unit(relevance), features, diversity, k=n)
    return cand[np.argsort(-relevance, kind="stable")[:n]]


def route(chosen, items, start, roads=None):
    """Step 3: the shortest visiting order of `chosen` from `start`.

    Returns the plan_day dict (without `ranked`); `chosen` is in recommendation order.
    """
    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) > MAX_PLACES:
        raise ValueError(f"at most {MAX_PLACES} places per route, got {len(chosen)}")
    lat, lon = items.lat.values, items.lon.values
    # np.r_[a, b] concatenates a scalar and an array into one 1-D array: the start
    # point becomes node 0, followed by the n chosen places
    pts_lat, pts_lon = np.r_[start[0], lat[chosen]], np.r_[start[1], lon[chosen]]
    straight = distance_matrix(pts_lat, pts_lon)
    routes = None
    if roads is not None and len(chosen) > 0:  # nothing to route: no street download
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
        "route_km": length,
        "rec_order_km": path_length(D, rec),
        "straight_km": path_length(straight, order),
        "legs_km": [float(D[a, b]) for a, b in zip(stops[:-1], stops[1:])],
        "path": draw(order),
        "rec_path": draw(rec),
    }


def plan_day(pred, filt, items, start, user=None, n=6, mask=None, gamma=GAMMA, tau=TAU, decay=DECAY, roads=None,
             visit=None, diversity=0.0, features=None):
    """Recommend n places near `start` and the shortest order to visit them.

    Args:
        pred: predicted rating (or any ranking score) for every item, e.g.
            model.score_all(user); with gamma = 0 it is used as is.
        filt: candidates.CandidateFilter (open, enough ratings, not yet visited).
        items: DataFrame from data.load_items().
        start: (lat, lon) where the day begins (user anchor, hotel, ...).
        user: user id whose visited places are excluded; None for a new visitor.
        n: number of places to visit.
        mask: optional boolean array over items, e.g. only bakeries, only open now.
        roads: None for straight-line routing, "osm" to download the walking
            network around the chosen places, a roads.RoadNetwork, or a
            function (lats, lons) -> RoadNetwork (e.g. to limit download time).
        visit: None for the fixed distance rule, or (model, features, columns)
            to rank with a learned visit model: a fitted
            models.visit.LogisticRegression, a models.visit.VisitFeatures and
            the feature columns it was trained on.
        diversity, features: MMR weight and item features (see recommend).

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
    would need a much larger network (experiments/walking.py: the accuracy is
    the same either way).
    """
    if n > MAX_PLACES:
        raise ValueError(f"n must be between 1 and {MAX_PLACES}, got {n}")
    chosen = recommend(pred, filt, items, start, user, n, mask, gamma, tau, decay, visit, diversity, features)
    return {**route(chosen, items, start, roads), "ranked": [int(c) for c in chosen]}


def plan_days(pred, filt, items, start, days, per_day=6, roads=None, **kw):
    """A multi-day plan: recommend days * per_day places, one neighborhood per day.

    The places are grouped by map position with k-means capped at per_day
    places per group (cluster.balanced_kmeans, k = days); each group is routed
    from `start` as one day, the group with the best-ranked place first. With
    fewer places than days * per_day the cap shrinks to spread them evenly, and
    a plan can have fewer days than asked for when there are very few places.
    The street network (roads = "osm" or a function) is fetched once around
    all chosen places and reused for every day. kw are passed to recommend
    (user, mask, visit, diversity, ...).

    Returns:
        dict with `days` (one route() dict per day, plus its `ranked` ids) and
        `ranked` (all ids in recommendation order).
    """
    chosen = recommend(pred, filt, items, start, n=days * per_day, **kw)
    if roads is not None and len(chosen) and not isinstance(roads, RoadNetwork):
        # one network for the whole trip: one download (or cache hit), not one per day
        lats = np.r_[start[0], items.lat.values[chosen]]
        lons = np.r_[start[1], items.lon.values[chosen]]
        roads = RoadNetwork.around(lats, lons) if isinstance(roads, str) else roads(lats, lons)
    if days <= 1 or len(chosen) <= 1:
        return {"days": [{**route(chosen, items, start, roads), "ranked": [int(c) for c in chosen]}],
                "ranked": [int(c) for c in chosen]}
    size = min(per_day, -(-len(chosen) // days))  # ceil: spread few places evenly over the days
    labels, _ = balanced_kmeans(to_km(items.lat.values[chosen], items.lon.values[chosen]), days, size)
    rank = {int(c): r for r, c in enumerate(chosen)}
    # np.unique(labels): each distinct label (day number) once, in ascending order
    groups = [chosen[labels == g] for g in np.unique(labels)]
    groups.sort(key=lambda g: min(rank[int(c)] for c in g))  # day 1 holds the top recommendation
    out = []
    for g in groups:
        g = sorted(g, key=lambda c: rank[int(c)])
        out.append({**route(g, items, start, roads), "ranked": [int(c) for c in g]})
    return {"days": out, "ranked": [int(c) for c in chosen]}
