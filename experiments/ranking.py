"""Top-N ranking per model, then distance re-ranking of the hybrid list (RQ1).

For 3,000 sampled warm test users with at least one relevant test place
(test rating >= 4): take the top 50 eligible candidates by predicted rating,
cut to the top 10, and score Precision / Recall / HitRate / NDCG at 5 and 10,
catalog coverage and the mean distance of the list from the user's anchor.
Then re-rank the hybrid candidates with each decay shape and distance scale
(figure R2: NDCG@10 against mean list distance).

Usage:  python experiments/ranking.py
Output: results/ranking.json
Targets (proposal 9): popularity P@10 0.0095, NDCG@10 0.0335; rating models
P@10 0.0005-0.0007; hybrid NDCG@10 0.0017 -> 0.0059 with exp decay, tau = 2 km,
mean list distance 6.3 -> 2.1 km.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tasteroute.candidates import CandidateFilter
from tasteroute.data import get_split, load_item_features, load_items, load_ratings, n_users_items
from tasteroute.geo import haversine, user_anchors
from tasteroute.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k
from tasteroute.models.baseline import BiasModel
from tasteroute.models.cbf import CBF
from tasteroute.models.hybrid import Hybrid
from tasteroute.models.mf import MF
from tasteroute.rerank import DECAYS, rerank

ROOT = Path(__file__).resolve().parents[1]
N_SAMPLE = 3000
SEED = 42
KS = (5, 10)
TAUS = [0.5, 1, 2, 3, 5, 10, 20, 50, 100, 1000]


def score_lists(lists, relevant, dists):
    """Average ranking metrics over users. lists/relevant/dists are per-user."""
    out = {}
    for k in KS:
        out[f"P@{k}"] = float(np.mean([precision_at_k(l, r, k) for l, r in zip(lists, relevant)]))
        out[f"R@{k}"] = float(np.mean([recall_at_k(l, r, k) for l, r in zip(lists, relevant)]))
        out[f"Hit@{k}"] = float(np.mean([hit_rate_at_k(l, r, k) for l, r in zip(lists, relevant)]))
        out[f"NDCG@{k}"] = float(np.mean([ndcg_at_k(l, r, k) for l, r in zip(lists, relevant)]))
    # users whose list is empty (every candidate filtered out) have no distance;
    # skip them, and store null rather than NaN if nobody has one
    means = [d.mean() for d in dists if len(d)]
    out["mean_dist_km"] = float(np.mean(means)) if means else None
    return out


def main():
    r = load_ratings()
    items = load_items()
    n_users, n_items = n_users_items(r)
    X, _ = load_item_features()
    train = get_split(r, "train")
    val = get_split(r, "val", warm_only=True)
    test = get_split(r, "test", warm_only=True)

    rel = test[test.stars >= 4].groupby("u").i.apply(set)
    rng = np.random.default_rng(SEED)
    users = np.sort(rng.choice(rel.index.values, size=N_SAMPLE, replace=False))
    relevant = [rel[u] for u in users]

    filt = CandidateFilter(items, train, seen=pd.concat([train, get_split(r, "val")]))
    anchors = user_anchors(train, items, n_users)
    lat, lon = items.lat.values, items.lon.values

    def dist(u, cand):
        return haversine(anchors[u, 0], anchors[u, 1], lat[cand], lon[cand])

    print("fitting models", flush=True)
    mf = MF(n_users, n_items, f=10, lam=0.1).fit(train, val)
    cbf = CBF(n_users, n_items, X, lam=30).fit(train)
    models = {
        "Bias": BiasModel(n_users, n_items, lam=5).fit(train),
        "MF": mf,
        "CBF": cbf,
        "Hybrid": Hybrid(mf, cbf, kappa=100).fit(train),
    }

    out = {"n_users": N_SAMPLE, "models": {}, "rerank": []}
    pop = filt.popularity
    lists = [filt.top_m(u, pop, m=10) for u in users]
    out["models"]["Popularity"] = score_lists(lists, relevant, [dist(u, l) for u, l in zip(users, lists)])
    out["models"]["Popularity"]["coverage"] = len(set(np.concatenate(lists))) / n_items

    hybrid_cands = None
    for name, m in models.items():
        cands = [filt.top_m(u, m.score_all(u)) for u in users]
        lists = [c[:10] for c in cands]
        out["models"][name] = score_lists(lists, relevant, [dist(u, l) for u, l in zip(users, lists)])
        out["models"][name]["coverage"] = len(set(np.concatenate(lists))) / n_items
        if name == "Hybrid":
            hybrid_cands = [(c, m.score_all(u)[c]) for u, c in zip(users, cands)]
    for name, res in out["models"].items():
        print(f"{name:<11} P@10 {res['P@10']:.4f}  NDCG@10 {res['NDCG@10']:.4f}  "
              f"dist {res['mean_dist_km']:.1f} km  coverage {res['coverage']:.4f}", flush=True)

    for decay in DECAYS:
        for tau in TAUS:
            lists, dists = [], []
            for u, (c, p) in zip(users, hybrid_cands):
                top = rerank(c, p, dist(u, c), tau, decay, k=10)
                lists.append(top)
                dists.append(dist(u, top))
            res = score_lists(lists, relevant, dists)
            out["rerank"].append({"decay": decay, "tau": tau, **res})
            print(f"rerank {decay:<8} tau={tau:<5} NDCG@10 {res['NDCG@10']:.4f}  P@10 {res['P@10']:.4f}  "
                  f"dist {res['mean_dist_km']:.1f} km", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ranking.json").write_text(json.dumps(out, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
