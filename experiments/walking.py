"""RQ1 and RQ3 with real walking distances instead of straight lines (issue #56).

Uses the cached Center City walking network (python -m tasteroute.roads).
Users' anchors must lie inside both CENTER_CITY_BOX and the network, at least
0.5 km from each boundary. Candidates are the top 200 by Hybrid blend score
inside both bounds, even when a larger cached network is reused.
    detour      walking km / straight km for every (anchor, candidate) pair
    RQ1         re-rank the candidates by exp(-d / tau) with d = straight or
                walking km; tau is chosen per distance type on validation
                users, then each is scored once on test (paired t-test)
    RQ3         top 6 of the straight re-rank; order them with Held-Karp on
                straight km and on walking km, and walk both orders: how much
                longer is the straight-optimal order on real streets?
Usage:  python experiments/walking.py
Output: results/walking.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from ranking import SEED, setup
from tasteroute.candidates import CandidateFilter, blend_popularity, to_unit
from tasteroute.data import get_split, load_ratings
from tasteroute.geo import distance_matrix, haversine
from tasteroute.metrics import ndcg_at_k, precision_at_k
from tasteroute.rerank import rerank
from tasteroute.roads import CENTER_CITY_BOX, KM_PER_DEG_LAT, RoadNetwork
from tasteroute.route import held_karp, path_length
ROOT = Path(__file__).resolve().parents[1]
POOL, GAMMA, N_ROUTE = 200, 5.0, 6
TAUS = [0.5, 1, 1.5, 2, 3, 5, 8, 12, 20]
N_VAL = 1500
EDGE_KM = 0.5


def inside_scope(net, lat, lon, margin_km=0.0):
    """Intersect fixed Center City bounds with the network's usable extent.

    download() may reuse a cache covering a larger area. Its node extent must
    not expand the experiment's population or candidate pool. User margins are
    measured from both bounds; the fixed box uses its own midpoint latitude.
    """
    north, south, east, west = CENTER_CITY_BOX
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    dlat = margin_km / KM_PER_DEG_LAT
    dlon = dlat / np.cos(np.radians((north + south) / 2))
    in_city = ((lat >= south + dlat) & (lat <= north - dlat)
               & (lon >= west + dlon) & (lon <= east - dlon))
    return in_city & net.contains(lat, lon, margin_km=margin_km)


def main():
    """Run this step and write its output (see the module docstring)."""
    t0 = time.time()
    ctx = setup()
    items, train, hybrid = ctx["items"], ctx["train"], ctx["models"]["Hybrid"]
    anchors, lat, lon = ctx["anchors"], ctx["items"].lat.values, ctx["items"].lon.values
    net = RoadNetwork.download(*CENTER_CITY_BOX)
    item_inside = inside_scope(net, lat, lon)
    print(f"network {len(net.lat):,} nodes; places inside {item_inside.sum():,}", flush=True)
    def pool(filt, u):
        """Blend-score top POOL inside the city and network, with straight and walking km from u's anchor."""
        score = blend_popularity(hybrid.score_all(u), filt.popularity, GAMMA)
        score = np.where(item_inside, score, -np.inf)
        cand = filt.top_m(u, score, m=POOL)
        d_s = haversine(anchors[u, 0], anchors[u, 1], lat[cand], lon[cand])
        d_w = net.distances_from(anchors[u, 0], anchors[u, 1], lat[cand], lon[cand])
        ok = np.isfinite(d_w)
        return cand[ok], score[cand][ok], d_s[ok], d_w[ok]
    def inside(users):
        return users[inside_scope(net, anchors[users, 0], anchors[users, 1], margin_km=EDGE_KM)]
    # ---- tau per distance type on validation users inside the network ----
    val = get_split(load_ratings(), "val", warm_only=True)
    rel_val = val[val.stars >= 4].groupby("u").i.apply(set)
    rng = np.random.default_rng(SEED + 4)
    val_users = inside(rel_val.index.values)
    val_users = rng.choice(val_users, size=min(N_VAL, len(val_users)), replace=False)
    val_filt = CandidateFilter(items, train, seen=train)
    val_pools = [pool(val_filt, u) for u in val_users]
    out = {"selection": [], "n_val_users": len(val_users)}
    best_tau = {}
    for kind, col in (("straight", 2), ("walking", 3)):
        for tau in TAUS:
            ndcg = np.mean([ndcg_at_k(rerank(p[0], 1 + 4 * to_unit(p[1]), p[col], tau, k=10), rel_val[u], 10)
                            for u, p in zip(val_users, val_pools)])
            out["selection"].append({"distance": kind, "tau": tau, "val_NDCG@10": float(ndcg)})
        best_tau[kind] = max((s for s in out["selection"] if s["distance"] == kind),
                             key=lambda s: s["val_NDCG@10"])["tau"]
    out["tau"] = best_tau
    print(f"validation: {len(val_users)} users, tau {best_tau} ({time.time() - t0:.0f}s)", flush=True)
    # ---- test users inside the network ----
    keep = inside_scope(net, anchors[ctx["users"], 0], anchors[ctx["users"], 1], margin_km=EDGE_KM)
    users = ctx["users"][keep]
    relevant = [r for r, k in zip(ctx["relevant"], keep) if k]
    filt = ctx["filt"]
    pools = [pool(filt, u) for u in users]
    print(f"test: {len(users)} users inside the network ({time.time() - t0:.0f}s)", flush=True)
    # detour factor over all (anchor, candidate) pairs farther than 300 m
    ratios = np.concatenate([p[3][p[2] > 0.3] / p[2][p[2] > 0.3] for p in pools] + [np.empty(0)])
    if len(ratios) == 0:
        raise RuntimeError("no (anchor, place) pairs inside the network; download it with python -m tasteroute.roads")
    out["detour"] = {"pairs": int(len(ratios)), "median": float(np.median(ratios)),
                     "p10": float(np.percentile(ratios, 10)), "p90": float(np.percentile(ratios, 90)),
                     "hist": np.histogram(ratios, bins=np.arange(1.0, 2.05, 0.05))[0].tolist()}
    print(f"detour walking/straight: median {out['detour']['median']:.2f} "
          f"(10-90%: {out['detour']['p10']:.2f}-{out['detour']['p90']:.2f})", flush=True)
    # ---- RQ1: straight vs walking re-rank, both measured in walking km ----
    out["rq1"], per_user = {}, {}
    for kind, col in (("straight", 2), ("walking", 3)):
        lists, walk_km, ndcg, prec = [], [], [], []
        for (cand, sc, ds, dw), rel in zip(pools, relevant):
            order = np.argsort(-(to_unit(sc) * np.exp(-(ds if col == 2 else dw) / best_tau[kind])), kind="stable")[:10]
            top = cand[order]
            lists.append(top)
            if len(order):  # a user whose pool is empty has no list distance
                walk_km.append(dw[order].mean())
            ndcg.append(ndcg_at_k(top, rel, 10))
            prec.append(precision_at_k(top, rel, 10))
        per_user[kind] = np.array(ndcg)
        out["rq1"][kind] = {"tau": best_tau[kind], "NDCG@10": float(np.mean(ndcg)), "P@10": float(np.mean(prec)),
                            "mean_walk_km": float(np.mean(walk_km)) if walk_km else None}
    t = stats.ttest_rel(per_user["walking"], per_user["straight"])
    out["rq1"]["walking_vs_straight"] = {"diff": float(np.mean(per_user["walking"] - per_user["straight"])),
                                         "p": float(t.pvalue)}
    for kind in ("straight", "walking"):
        r = out["rq1"][kind]
        print(f"RQ1 re-rank by {kind:<8} tau={r['tau']:<4} NDCG@10 {r['NDCG@10']:.4f}  P@10 {r['P@10']:.4f}  "
              f"list walking distance {r['mean_walk_km'] if r['mean_walk_km'] is None else round(r['mean_walk_km'], 2)} km",
              flush=True)
    print(f"RQ1 walking vs straight NDCG@10 {out['rq1']['walking_vs_straight']['diff']:+.4f} "
          f"p = {out['rq1']['walking_vs_straight']['p']:.3g}", flush=True)
    # ---- RQ3: order the top 6 by straight vs walking km, walk both ----
    rows = []
    for u, (cand, sc, ds, dw) in zip(users, pools):
        if len(cand) < N_ROUTE:
            continue
        top = cand[np.argsort(-(to_unit(sc) * np.exp(-ds / best_tau["straight"])), kind="stable")[:N_ROUTE]]
        pts_lat, pts_lon = np.r_[anchors[u, 0], lat[top]], np.r_[anchors[u, 1], lon[top]]
        r = net.routes(pts_lat, pts_lon)
        if not r.connected.all():
            continue
        order_s, _ = held_karp(distance_matrix(pts_lat, pts_lon))
        order_w, best = held_karp(r.D)
        rows.append({"u": int(u), "walk_optimal": best, "walk_of_straight_order": path_length(r.D, order_s),
                     "walk_of_rec_order": path_length(r.D, range(1, N_ROUTE + 1)),
                     "straight_optimal": path_length(distance_matrix(pts_lat, pts_lon), order_s)})
    if not rows:
        raise RuntimeError("no test user has 6 connected places inside the network")
    extra = np.array([x["walk_of_straight_order"] / x["walk_optimal"] - 1 for x in rows])
    out["rq3"] = {
        "users": len(rows), "same_order_share": float(np.mean(extra < 1e-9)),
        "extra_mean": float(extra.mean()), "extra_p90": float(np.percentile(extra, 90)),
        "extra_max": float(extra.max()),
        "walk_vs_straight_length_median": float(np.median([x["walk_optimal"] / x["straight_optimal"] for x in rows])),
        "rec_order_extra_median": float(np.median([x["walk_of_rec_order"] / x["walk_optimal"] - 1 for x in rows])),
        "extra_hist": np.histogram(extra * 100, bins=[0, 1e-9, 2, 5, 10, 20, 100])[0].tolist(),
        "per_user": rows,
    }
    q = out["rq3"]
    print(f"RQ3 {q['users']} users: straight-optimal order is also walking-optimal for {q['same_order_share']:.1%}; "
          f"mean extra walking {q['extra_mean']:.1%} (90th pct {q['extra_p90']:.1%}, max {q['extra_max']:.1%}); "
          f"optimal walk / straight length median {q['walk_vs_straight_length_median']:.2f}; "
          f"recommendation order walks {q['rec_order_extra_median']:.1%} more (median)", flush=True)
    out["n_test_users"] = len(users)
    out["seconds"] = round(time.time() - t0)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "walking.json").write_text(json.dumps(out, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
