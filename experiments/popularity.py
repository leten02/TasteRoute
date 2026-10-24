"""Popularity-aware candidate generation (code plan item 4, proposal 9 open problem).

Pure popularity beats every rating model on Precision@10 because the models are
trained to predict ratings, not visits. Two fixes for the Hybrid model:
    blend      score = r_hat + gamma * log(1 + popularity)
    two-stage  the `pool` most popular places, re-ordered by r_hat
Popularity alone, with and without distance, is the reference both must beat.
Both are tuned on validation (3,000 users, relevant = val rating >= 4, only
train items excluded), then the best of each is scored once on the test users
of experiments/ranking.py, with and without distance re-ranking.

Usage:  python experiments/popularity.py
Output: results/popularity.json
Goal (code plan item 4): P@10 at least popularity's, with higher catalog coverage.
"""
import json
from pathlib import Path

import numpy as np

from ranking import N_SAMPLE, SEED, score_lists, setup
from tasteroute.candidates import CandidateFilter, blend_popularity, to_unit, two_stage
from tasteroute.data import get_split, load_ratings
from tasteroute.rerank import rerank

ROOT = Path(__file__).resolve().parents[1]
GAMMAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10, 20]
POOLS = [20, 50, 100, 200, 500]
TAU, DECAY = 2.0, "exp"


def run(users, relevant, filt, dist, model, method, param, n_items, distance=False):
    """Score one candidate method for these users; returns averaged ranking metrics."""
    lists, dists = [], []
    for u in users:
        pred = model.score_all(u)
        if method == "popularity":  # no model: popularity alone (reference)
            score = filt.popularity
            cand = filt.top_m(u, score)
        elif method == "blend":
            score = blend_popularity(pred, filt.popularity, param)
            cand = filt.top_m(u, score)
        else:
            cand = two_stage(filt, u, pred, pool=param)[:50]
            score = pred
        if distance:
            top = rerank(cand, 1 + 4 * to_unit(score[cand]), dist(u, cand), TAU, DECAY, k=10)
        else:
            top = cand[:10]
        lists.append(top)
        dists.append(dist(u, top))
    res = score_lists(lists, relevant, dists)
    res["coverage"] = len(set(np.concatenate(lists))) / n_items
    return res


def main():
    """Run this step and write its output (see the module docstring)."""
    ctx = setup()
    hybrid, dist, items, train = ctx["models"]["Hybrid"], ctx["dist"], ctx["items"], ctx["train"]
    n_items = ctx["n_items"]

    # validation users: relevant = val rating >= 4; only training items are "seen"
    val = get_split(load_ratings(), "val", warm_only=True)
    rel_val = val[val.stars >= 4].groupby("u").i.apply(set)
    rng = np.random.default_rng(SEED + 1)
    val_users = np.sort(rng.choice(rel_val.index.values, size=N_SAMPLE, replace=False))
    val_relevant = [rel_val[u] for u in val_users]
    val_filt = CandidateFilter(items, train, seen=train)

    out = {"val": [], "test": {}}
    for method, params in [("blend", GAMMAS), ("two_stage", POOLS)]:
        for p in params:
            res = run(val_users, val_relevant, val_filt, dist, hybrid, method, p, n_items)
            out["val"].append({"method": method, "param": p, **res})
            print(f"val {method:<9} {p:<5} P@10 {res['P@10']:.4f}  NDCG@10 {res['NDCG@10']:.4f}  "
                  f"coverage {res['coverage']:.4f}", flush=True)

    users, relevant, filt = ctx["users"], ctx["relevant"], ctx["filt"]
    for method in ("popularity", "blend", "two_stage"):
        if method == "popularity":
            best = {"param": "-"}
        else:
            best = max((v for v in out["val"] if v["method"] == method), key=lambda v: v["NDCG@10"])
        for distance in (False, True):
            key = f"{method}({best['param']})" + (" + distance" if distance else "")
            out["test"][key] = run(users, relevant, filt, dist, hybrid, method, best["param"], n_items, distance)
            res = out["test"][key]
            print(f"test {key:<28} P@10 {res['P@10']:.4f}  NDCG@10 {res['NDCG@10']:.4f}  "
                  f"dist {res['mean_dist_km']:.1f} km  coverage {res['coverage']:.4f}", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "popularity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
