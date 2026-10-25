"""Re-evaluate the main results on a time-ordered split (issue #67).

The default split shuffles each user's ratings, so 88% of warm users have a
training rating that is newer than one of their test ratings. Here each warm
user's oldest ~70% of ratings train the models, the next ~15% validate and the
newest ~15% test (data.temporal_split): learn the past, predict the future.

Rating prediction: Global mean, Bias, MF, CBF and Hybrid. The Bias lambda, CBF
lambda_c and Hybrid kappa are re-chosen on the temporal validation set; MF keeps
f = 10, lambda = 0.1 and early-stops on the temporal validation set.
Ranking (3,000 sampled warm test users, relevant = test rating >= 4): popularity,
Hybrid, Hybrid + distance (exp, tau = 2 km) and the popularity-blended Hybrid
with and without distance; gamma is re-chosen on temporal validation users the
same way as experiments/popularity.py.
Every temporal number is stored next to the random-split number from
results/model_compare.json, results/ranking.json and results/popularity.json.

Usage:  python experiments/temporal.py   (after compare_models, ranking, popularity)
Output: results/temporal.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from compare_models import SEED as BOOT_SEED
from compare_models import bootstrap_rmse, per_user_sse
from popularity import DECAY, GAMMAS, TAU, run
from ranking import N_SAMPLE, SEED, score_lists
from tasteroute.candidates import CandidateFilter
from tasteroute.data import get_split, load_item_features, load_items, load_ratings, n_users_items, temporal_split
from tasteroute.geo import haversine, user_anchors
from tasteroute.metrics import mae, rmse
from tasteroute.models.baseline import BiasModel, GlobalMean
from tasteroute.models.cbf import CBF
from tasteroute.models.hybrid import Hybrid
from tasteroute.models.mf import MF
from tasteroute.rerank import rerank

ROOT = Path(__file__).resolve().parents[1]
BIAS_LAMS = [1, 5, 10, 25]
CBF_LAMS = [3, 10, 30, 100]
KAPPAS = [1, 10, 100, 1000]
METRICS = ("P@10", "NDCG@10", "mean_dist_km", "coverage")


def leakage_share(ratings):
    """Share of warm users with a training rating newer than one of their test ratings."""
    warm = ratings[ratings.cold_m == 0]
    newest_train = warm[warm.split == "train"].groupby("u").date.max()
    oldest_test = warm[warm.split == "test"].groupby("u").date.min()
    return float((newest_train > oldest_test.reindex(newest_train.index)).mean())


def split_summary(ratings):
    """Split sizes (warm users), the date range of each split and the leakage share."""
    warm = ratings[ratings.cold_m == 0]
    out = {"leakage_share": leakage_share(ratings)}
    for name in ("train", "val", "test"):
        d = warm[warm.split == name].date
        out[name] = {"ratings": int(len(d)), "median_date": str(d.median().date())}
    return out


def pick(grid, fit_score):
    """Return (best value, {value: val RMSE}) for a 1-D validation grid."""
    scores = {v: fit_score(v) for v in grid}
    return min(scores, key=scores.get), scores


def rating_models(r):
    """Fit the rating models on the temporal split; returns (models, selected, val_grid)."""
    n_users, n_items = n_users_items(r)
    X, _ = load_item_features()
    train = get_split(r, "train")
    val = get_split(r, "val", warm_only=True)

    def val_rmse(m):
        return rmse(val.stars.values, m.predict(val.u.values, val.i.values))

    lam_b, grid_b = pick(BIAS_LAMS, lambda v: val_rmse(BiasModel(n_users, n_items, lam=v).fit(train)))
    lam_c, grid_c = pick(CBF_LAMS, lambda v: val_rmse(CBF(n_users, n_items, X, lam=v).fit(train)))
    mf = MF(n_users, n_items, f=10, lam=0.1).fit(train, val)
    cbf = CBF(n_users, n_items, X, lam=lam_c).fit(train)
    kappa, grid_k = pick(KAPPAS, lambda v: val_rmse(Hybrid(mf, cbf, kappa=v).fit(train)))
    models = {
        "Global mean": GlobalMean(n_users, n_items).fit(train),
        "Bias": BiasModel(n_users, n_items, lam=lam_b).fit(train),
        "MF": mf,
        "CBF": cbf,
        "Hybrid": Hybrid(mf, cbf, kappa=kappa).fit(train),
    }
    selected = {"Bias lambda": lam_b, "CBF lambda_c": lam_c, "Hybrid kappa": kappa, "MF": "f = 10, lambda = 0.1"}
    val_grid = {"Bias lambda": grid_b, "CBF lambda_c": grid_c, "Hybrid kappa": grid_k}
    return models, selected, val_grid


def rating_results(r, models):
    """Warm test RMSE, MAE, 95% bootstrap interval and paired t-test against Bias."""
    test = get_split(r, "test", warm_only=True)
    rng = np.random.default_rng(BOOT_SEED)
    sse_bias, _ = per_user_sse(test, models["Bias"].predict(test.u.values, test.i.values))
    out = {}
    for name, m in models.items():
        pred = m.predict(test.u.values, test.i.values)
        sse, cnt = per_user_sse(test, pred)
        users = np.flatnonzero(cnt)
        res = {"rmse": rmse(test.stars.values, pred), "mae": mae(test.stars.values, pred),
               "rmse_ci95": bootstrap_rmse(sse, cnt, rng)}
        if name != "Bias":
            # stats.ttest_rel(a, b): paired t-test on the same users' mean squared error
            p = float(stats.ttest_rel(sse[users] / cnt[users], sse_bias[users] / cnt[users]).pvalue)
            res["vs_bias_p"] = p if np.isfinite(p) else None
        out[name] = res
    return out


def ranking_results(r, hybrid):
    """Ranking metrics on the temporal split; gamma re-chosen on temporal validation users."""
    items = load_items()
    n_users, n_items = n_users_items(r)
    train = get_split(r, "train")
    anchors = user_anchors(train, items, n_users)
    lat, lon = items.lat.values, items.lon.values

    def dist(u, cand):
        """Distance (km) from user u's anchor to each place in cand."""
        return haversine(anchors[u, 0], anchors[u, 1], lat[cand], lon[cand])

    # validation users pick gamma: relevant = val rating >= 4, only train items are "seen"
    val = get_split(r, "val", warm_only=True)
    rel_val = val[val.stars >= 4].groupby("u").i.apply(set)
    rng = np.random.default_rng(SEED + 1)
    val_users = np.sort(rng.choice(rel_val.index.values, size=N_SAMPLE, replace=False))
    val_filt = CandidateFilter(items, train, seen=train)
    val_ndcg = {}
    for g in GAMMAS:
        res = run(val_users, [rel_val[u] for u in val_users], val_filt, dist, hybrid, "blend", g, n_items)
        val_ndcg[g] = res["NDCG@10"]
    gamma = max(val_ndcg, key=val_ndcg.get)

    test = get_split(r, "test", warm_only=True)
    rel = test[test.stars >= 4].groupby("u").i.apply(set)
    rng = np.random.default_rng(SEED)
    users = np.sort(rng.choice(rel.index.values, size=N_SAMPLE, replace=False))
    relevant = [rel[u] for u in users]
    filt = CandidateFilter(items, train, seen=pd.concat([train, get_split(r, "val")]))

    out = {}
    for distance in (False, True):
        key = "Popularity" + (" + distance" if distance else "")
        out[key] = run(users, relevant, filt, dist, hybrid, "popularity", None, n_items, distance)

    # the hybrid alone and with distance, exactly as in experiments/ranking.py
    cands = [filt.top_m(u, hybrid.score_all(u)) for u in users]
    lists = [c[:10] for c in cands]
    out["Hybrid"] = score_lists(lists, relevant, [dist(u, l) for u, l in zip(users, lists)])
    out["Hybrid"]["coverage"] = len(set(np.concatenate(lists))) / n_items
    lists = [rerank(c, hybrid.score_all(u)[c], dist(u, c), TAU, DECAY, k=10) for u, c in zip(users, cands)]
    out["Hybrid + distance"] = score_lists(lists, relevant, [dist(u, l) for u, l in zip(users, lists)])
    out["Hybrid + distance"]["coverage"] = len(set(np.concatenate(lists))) / n_items

    for distance in (False, True):
        key = "Hybrid blend" + (" + distance" if distance else "")
        out[key] = run(users, relevant, filt, dist, hybrid, "blend", gamma, n_items, distance)
    return out, gamma, val_ndcg


def random_reference():
    """The same numbers on the random split, read from the earlier experiments."""
    mc = json.loads((ROOT / "results" / "model_compare.json").read_text())["warm"]
    rk = json.loads((ROOT / "results" / "ranking.json").read_text())
    pop = json.loads((ROOT / "results" / "popularity.json").read_text())
    gamma = max((v for v in pop["val"] if v["method"] == "blend"), key=lambda v: v["NDCG@10"])["param"]
    rer = next(x for x in rk["rerank"] if x["decay"] == DECAY and x["tau"] == TAU)
    ranking = {
        "Popularity": pop["test"]["popularity(-)"],
        "Popularity + distance": pop["test"]["popularity(-) + distance"],
        "Hybrid": rk["models"]["Hybrid"],
        "Hybrid + distance": rer,
        "Hybrid blend": pop["test"][f"blend({gamma})"],
        "Hybrid blend + distance": pop["test"][f"blend({gamma}) + distance"],
    }
    return mc, ranking, gamma


def main():
    """Run this step and write its output (see the module docstring)."""
    r = load_ratings()
    rt = temporal_split(r)
    out = {"split": {"random": split_summary(r), "temporal": split_summary(rt)}}
    print(f"leakage share: random {out['split']['random']['leakage_share']:.3f}, "
          f"temporal {out['split']['temporal']['leakage_share']:.3f}", flush=True)

    print("fitting models on the temporal split", flush=True)
    models, selected, val_grid = rating_models(rt)
    rand_rating, rand_ranking, rand_gamma = random_reference()
    temp_rating = rating_results(rt, models)
    out["rating"] = {"selected": selected, "val_grid": val_grid,
                     "test": {n: {"random": rand_rating[n], "temporal": temp_rating[n]} for n in temp_rating}}
    for n, v in out["rating"]["test"].items():
        print(f"{n:<12} RMSE random {v['random']['rmse']:.4f}  temporal {v['temporal']['rmse']:.4f}", flush=True)

    temp_ranking, gamma, val_ndcg = ranking_results(rt, models["Hybrid"])
    out["ranking"] = {
        "n_users": N_SAMPLE, "gamma": {"random": rand_gamma, "temporal": gamma}, "val_ndcg_by_gamma": val_ndcg,
        "methods": {k: {"random": {m: rand_ranking[k].get(m) for m in METRICS},
                        "temporal": {m: temp_ranking[k].get(m) for m in METRICS}} for k in temp_ranking},
    }
    for k, v in out["ranking"]["methods"].items():
        print(f"{k:<24} P@10 {v['random']['P@10']:.4f} -> {v['temporal']['P@10']:.4f}  "
              f"NDCG@10 {v['random']['NDCG@10']:.4f} -> {v['temporal']['NDCG@10']:.4f}  "
              f"dist {v['temporal']['mean_dist_km']:.1f} km", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    # allow_nan=False: fail loudly instead of writing NaN, which is not valid JSON
    (ROOT / "results" / "temporal.json").write_text(json.dumps(out, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
