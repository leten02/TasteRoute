"""BPR: learn the ranking of places directly, then compare with popularity (issue #52).

Selection (validation only): 2,000 users with a relevant validation place
(val rating >= 4), only training places excluded; each configuration trains
with early stopping on their NDCG@10 (patience 5). Grid: learning rate, factors
f, L2 lam, and what counts as a positive (every training visit, or only visits
rated >= 4).

Test (once, the 3,000 users of experiments/ranking.py), top 10 per user:
    popularity, Hybrid blend (gamma = 5), BPR, switching   without distance
    the same four re-ranked by exp(-d / 2 km) over their top 200 candidates
Switching hybrid: BPR for users with at least T training ratings, popularity
for the rest; T is chosen on the validation users.

Usage:  python experiments/bpr.py
Output: results/bpr.json, data/processed/bpr_model.npz (best model, for the dashboard)
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from ranking import N_SAMPLE, SEED, score_lists, setup
from tasteroute.candidates import CandidateFilter, blend_popularity, to_unit
from tasteroute.data import get_split, load_ratings
from tasteroute.metrics import ndcg_at_k
from tasteroute.models.bpr import BPR
from tasteroute.rerank import rerank

ROOT = Path(__file__).resolve().parents[1]
N_VAL = 2000
LRS = [0.01, 0.05]
FS = [32, 64, 128]
LAMBDAS = [1e-4, 1e-3]
POSITIVES = ["all", "liked"]  # every training visit / only stars >= 4
POOL, TAU, GAMMA = 200, 2.0, 5.0
THRESHOLDS = [1, 5, 10, 15, 20, 30, 50]


def top10(filt, u, score, dist=None):
    """Top 10 eligible unseen places; with dist, re-rank the top POOL by distance first."""
    if dist is None:
        return filt.top_m(u, score, m=10)
    cand = filt.top_m(u, score, m=POOL)
    return rerank(cand, 1 + 4 * to_unit(score[cand]), dist(u, cand), TAU, k=10)


def main():
    """Run this step and write its output (see the module docstring)."""
    ctx = setup()
    items, train, filt, dist, n_items = ctx["items"], ctx["train"], ctx["filt"], ctx["dist"], ctx["n_items"]
    n_users = len(ctx["anchors"])
    hybrid = ctx["models"]["Hybrid"]

    val = get_split(load_ratings(), "val", warm_only=True)
    rel_val = val[val.stars >= 4].groupby("u").i.apply(set)
    rng = np.random.default_rng(SEED + 3)
    val_users = rng.choice(rel_val.index.values, size=N_VAL, replace=False)
    val_filt = CandidateFilter(items, train, seen=train)

    def val_ndcg(model):
        return np.mean([ndcg_at_k(val_filt.top_m(u, model.score_all(u), m=10), rel_val[u], 10) for u in val_users])

    out = {"grid": [], "test": {}}
    best = None
    for positives in POSITIVES:
        pos = train if positives == "all" else train[train.stars >= 4]
        for lr in LRS:
            for f in FS:
                for lam in LAMBDAS:
                    t = time.time()
                    m = BPR(n_users, n_items, f=f, lr=lr, lam=lam, patience=5).fit(pos, val_score=val_ndcg)
                    rec = {"positives": positives, "lr": lr, "f": f, "lam": lam,
                           "val_NDCG@10": max(h["val"] for h in m.history),
                           "best_epoch": m.best_epoch, "epochs": len(m.history)}
                    out["grid"].append(rec)
                    print(f"{positives:<5} lr={lr:<5} f={f:<3} lam={lam:<6} val NDCG@10 {rec['val_NDCG@10']:.4f} "
                          f"epoch {m.best_epoch}/{len(m.history)} ({time.time() - t:.0f}s)", flush=True)
                    if best is None or rec["val_NDCG@10"] > best[0]["val_NDCG@10"]:
                        best = (rec, m)
    rec, bpr = best
    out["best"] = rec
    out["curve"] = bpr.history
    print(f"chosen {rec}", flush=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    bpr.save(ROOT / "data" / "processed" / "bpr_model.npz")

    # validation NDCG of popularity on the same users, for reference
    out["val_popularity_NDCG@10"] = float(np.mean(
        [ndcg_at_k(val_filt.top_m(u, val_filt.popularity, m=10), rel_val[u], 10) for u in val_users]))

    # switching threshold T on validation: BPR if the user has >= T training ratings
    n_train_all = np.bincount(train.u.values, minlength=n_users)
    val_bpr = np.array([ndcg_at_k(val_filt.top_m(u, bpr.score_all(u), m=10), rel_val[u], 10) for u in val_users])
    val_pop = np.array([ndcg_at_k(val_filt.top_m(u, val_filt.popularity, m=10), rel_val[u], 10) for u in val_users])
    out["switch_selection"] = []
    for T in THRESHOLDS:
        use = n_train_all[val_users] >= T
        out["switch_selection"].append({"T": T, "val_NDCG@10": float(np.where(use, val_bpr, val_pop).mean())})
    T_best = max(out["switch_selection"], key=lambda r: r["val_NDCG@10"])["T"]
    out["switch_T"] = T_best
    print(f"switching threshold chosen on validation: T = {T_best} "
          f"{[(r['T'], round(r['val_NDCG@10'], 4)) for r in out['switch_selection']]}", flush=True)

    users, relevant = ctx["users"], ctx["relevant"]
    scorers = {
        "Popularity": lambda u: filt.popularity,
        "Hybrid blend": lambda u: blend_popularity(hybrid.score_all(u), filt.popularity, GAMMA),
        "BPR": bpr.score_all,
        "Switching": lambda u: bpr.score_all(u) if n_train_all[u] >= T_best else filt.popularity,
    }
    per_user = {}
    for name, score in scorers.items():
        for with_dist in (False, True):
            key = name + (" + distance" if with_dist else "")
            lists = [top10(filt, u, score(u), dist if with_dist else None) for u in users]
            res = score_lists(lists, relevant, [dist(u, top) for u, top in zip(users, lists)])
            res["coverage"] = len(set(np.concatenate(lists))) / n_items
            out["test"][key] = res
            per_user[key] = np.array([ndcg_at_k(top, r, 10) for top, r in zip(lists, relevant)])
            print(f"test {key:<26} P@10 {res['P@10']:.4f}  NDCG@10 {res['NDCG@10']:.4f}  "
                  f"dist {res['mean_dist_km']:.1f} km  coverage {res['coverage']:.4f}", flush=True)

    # stats.ttest_rel: paired t-test on the same users' NDCG@10
    pairs = [("BPR", "Popularity"), ("BPR + distance", "Popularity + distance"),
             ("BPR + distance", "Hybrid blend + distance"), ("Switching", "Popularity"),
             ("Switching + distance", "Popularity + distance"), ("Switching + distance", "Hybrid blend + distance")]
    out["paired"] = {f"{a} vs {b}": {"diff": float(np.mean(per_user[a] - per_user[b])),
                                     "p": float(stats.ttest_rel(per_user[a], per_user[b]).pvalue)}
                     for a, b in pairs}
    for k, v in out["paired"].items():
        print(f"{k:<44} NDCG@10 diff {v['diff']:+.4f}  p = {v['p']:.3g}", flush=True)

    # does personalization pay off for active users? split test users by training ratings
    n_train = n_train_all[users]
    out["by_activity"] = []
    for lo, hi in [(1, 5), (6, 20), (21, 10**6)]:
        g = (n_train >= lo) & (n_train <= hi)
        a, b = per_user["BPR"][g], per_user["Popularity"][g]
        row = {"train_ratings": f"{lo}-{hi}" if hi < 10**6 else f"{lo}+", "users": int(g.sum()),
               "BPR": float(a.mean()), "Popularity": float(b.mean()),
               "p": float(stats.ttest_rel(a, b).pvalue)}
        out["by_activity"].append(row)
        print(f"train ratings {row['train_ratings']:<6} users {row['users']:<5} NDCG@10 BPR {row['BPR']:.4f} "
              f"vs popularity {row['Popularity']:.4f}  p = {row['p']:.3g}", flush=True)
    out["n_users"] = N_SAMPLE

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "bpr.json").write_text(json.dumps(out, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
