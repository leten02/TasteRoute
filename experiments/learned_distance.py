"""Learn the distance trade-off with logistic regression instead of a fixed rule (issue #48).

Fixed rule (experiments/popularity.py, best config): candidates = top 50 by
blend score r_hat + 5 log(1 + pop), re-ranked by rescaled score x exp(-d / 2 km).
Learned: candidates = top POOL by the same blend score, ranked by the logit of
models/visit.py's LogisticRegression on [log1p(d), d, log1p(d / rho_u), r_hat, log1p(pop)].

For a fair comparison the fixed rule's tau is also tuned on the same held-out
validation users, on the same pool of POOL candidates.

Training labels come from validation (relevant = val rating >= 4), so test is
never touched while fitting: rows are (user, candidate) pairs of N_TRAIN users
with at least one relevant place in their pool. lambda and the feature set are
chosen on 20% of those users held out; the chosen model is refit on all of them
and scored once on the 3,000 test users of experiments/ranking.py.

Usage:  python experiments/learned_distance.py
Output: results/learned_distance.json (metrics, coefficients, fitted model for the dashboard)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from ranking import SEED, score_lists, setup
from tasteroute.candidates import CandidateFilter, blend_popularity, to_unit
from tasteroute.data import get_split, load_ratings
from tasteroute.metrics import ndcg_at_k
from tasteroute.models.visit import FEATURES, LogisticRegression, VisitFeatures, rank_learned
from tasteroute.rerank import rerank

ROOT = Path(__file__).resolve().parents[1]
N_TRAIN = 6000
POOL = 200
GAMMA, TAU = 5.0, 2.0
LAMBDAS = [0, 1e-4, 1e-3, 1e-2, 1e-1]
TAUS = [0.5, 1, 1.5, 2, 3, 5]
FEATURE_SETS = {"no_radius": [0, 1, 3, 4], "with_radius": [0, 1, 2, 3, 4]}
CURVE_KM = np.linspace(0, 10, 51)


def pools(users, filt, model, feats, m):
    """Per user: blend-score candidate pool, its features and its distances."""
    out = []
    for u in users:
        pred = model.score_all(u)
        score = blend_popularity(pred, filt.popularity, GAMMA)
        cand = filt.top_m(u, score, m=m)
        out.append((u, cand, score[cand], feats.build(u, cand, pred[cand]), feats.distance(u, cand)))
    return out


def evaluate(lists, relevant, feats, users):
    """Ranking metrics plus per-user NDCG@10 (for the paired test)."""
    res = score_lists(lists, relevant, [feats.distance(u, top) for u, top in zip(users, lists)])
    per_user = np.array([ndcg_at_k(top, rel, 10) for top, rel in zip(lists, relevant)])
    return res, per_user


def main():
    """Run this step and write its output (see the module docstring)."""
    ctx = setup()
    hybrid, items, train, filt = ctx["models"]["Hybrid"], ctx["items"], ctx["train"], ctx["filt"]
    feats = VisitFeatures(items, train, ctx["anchors"], filt.popularity)

    # ---- training data from validation labels (only training items are "seen") ----
    val = get_split(load_ratings(), "val", warm_only=True)
    rel_val = val[val.stars >= 4].groupby("u").i.apply(set)
    rng = np.random.default_rng(SEED + 2)
    tr_users = rng.choice(rel_val.index.values, size=N_TRAIN, replace=False)
    val_filt = CandidateFilter(items, train, seen=train)
    print(f"building pools for {N_TRAIN} training users", flush=True)
    tr_pools = pools(tr_users, val_filt, hybrid, feats, POOL)
    tr_rel = [rel_val[u] for u in tr_users]

    holdout = np.zeros(N_TRAIN, dtype=bool)
    holdout[rng.choice(N_TRAIN, size=N_TRAIN // 5, replace=False)] = True

    def stack(mask):
        """Rows of users in mask that have a relevant place in their pool."""
        X, y = [], []
        for keep, (u, cand, _, Xu, _), rel in zip(mask, tr_pools, tr_rel):
            yu = np.isin(cand, list(rel)).astype(float)
            if keep and yu.any():
                X.append(Xu)
                y.append(yu)
        return np.vstack(X), np.concatenate(y)

    X_fit, y_fit = stack(~holdout)
    print(f"fit rows {len(y_fit):,}, positives {int(y_fit.sum()):,}", flush=True)

    out = {"pool": POOL, "features": list(FEATURES), "selection": [], "test": {}}
    best = None
    ho = [(p, r) for p, r, h in zip(tr_pools, tr_rel, holdout) if h]
    for fs_name, cols in FEATURE_SETS.items():
        for lam in LAMBDAS:
            m = LogisticRegression(lam=lam).fit(X_fit[:, cols], y_fit)
            lists = [rank_learned(m, Xu, cand, columns=cols) for (_, cand, _, Xu, _), _ in ho]
            ndcg = float(np.mean([ndcg_at_k(top, r, 10) for top, (_, r) in zip(lists, ho)]))
            out["selection"].append({"features": fs_name, "lam": lam, "holdout_NDCG@10": ndcg,
                                     "iters": len(m.history)})
            print(f"holdout {fs_name:<11} lam={lam:<7} NDCG@10 {ndcg:.4f} ({len(m.history)} iters)", flush=True)
            if best is None or ndcg > best[0]:
                best = (ndcg, fs_name, lam)
    _, best_fs, best_lam = best
    print(f"chosen: {best_fs}, lam={best_lam}", flush=True)

    # the fixed rule gets the same chance: tune tau on the same held-out users and pool
    best_tau = None
    for tau in TAUS:
        lists = [rerank(c, 1 + 4 * to_unit(s), d, tau, k=10) for (_, c, s, _, d), _ in ho]
        ndcg = float(np.mean([ndcg_at_k(top, r, 10) for top, (_, r) in zip(lists, ho)]))
        out["selection"].append({"features": "fixed rule", "tau": tau, "holdout_NDCG@10": ndcg})
        print(f"holdout fixed rule  tau={tau:<4} NDCG@10 {ndcg:.4f}", flush=True)
        if best_tau is None or ndcg > best_tau[0]:
            best_tau = (ndcg, tau)
    best_tau = best_tau[1]

    # ---- refit both feature sets on all training users (chosen lambda) ----
    X_all, y_all = stack(np.ones(N_TRAIN, dtype=bool))
    fitted = {fs: LogisticRegression(lam=best_lam).fit(X_all[:, cols], y_all) for fs, cols in FEATURE_SETS.items()}

    # ---- one test evaluation on the ranking.py users ----
    users, relevant = ctx["users"], ctx["relevant"]
    te_pools = pools(users, filt, hybrid, feats, POOL)
    pop_lists = []
    for u in users:
        cand = filt.top_m(u, filt.popularity)
        pop_lists.append(rerank(cand, 1 + 4 * to_unit(filt.popularity[cand]), feats.distance(u, cand), TAU, k=10))
    methods = {
        "popularity + fixed distance": pop_lists,
        "blend + fixed distance (top 50)": [
            rerank(c[:50], 1 + 4 * to_unit(s[:50]), d[:50], TAU, k=10) for _, c, s, _, d in te_pools],
        f"blend + fixed distance (top {POOL}, tau={best_tau})": [
            rerank(c, 1 + 4 * to_unit(s), d, best_tau, k=10) for _, c, s, _, d in te_pools],
        "learned (no personal radius)": [
            rank_learned(fitted["no_radius"], Xu, c, columns=FEATURE_SETS["no_radius"]) for _, c, _, Xu, _ in te_pools],
        "learned (with personal radius)": [
            rank_learned(fitted["with_radius"], Xu, c, columns=FEATURE_SETS["with_radius"]) for _, c, _, Xu, _ in te_pools],
    }
    per_user = {}
    for name, lists in methods.items():
        res, per_user[name] = evaluate(lists, relevant, feats, users)
        res["coverage"] = len(set(np.concatenate(lists))) / ctx["n_items"]
        out["test"][name] = res
        print(f"test {name:<34} P@10 {res['P@10']:.4f}  NDCG@10 {res['NDCG@10']:.4f}  "
              f"dist {res['mean_dist_km']:.1f} km  coverage {res['coverage']:.4f}", flush=True)

    # stats.ttest_rel(a, b): paired t-test on the same users' NDCG@10 under two methods
    out["paired_vs_fixed"] = {}
    for ref_name in ("blend + fixed distance (top 50)", f"blend + fixed distance (top {POOL}, tau={best_tau})"):
        out["paired_vs_fixed"][ref_name] = {
            name: {"diff": float(np.mean(v - per_user[ref_name])), "p": float(stats.ttest_rel(v, per_user[ref_name]).pvalue)}
            for name, v in per_user.items() if name.startswith("learned")}
    print("paired t-test p vs fixed rule:", out["paired_vs_fixed"], flush=True)

    # ---- interpretation of the chosen model ----
    chosen = fitted[best_fs]
    cols = FEATURE_SETS[best_fs]
    w, b = chosen.coef()
    names = [FEATURES[c] for c in cols]
    coef = dict(zip(names, w.tolist()))
    rho = feats.default_radius
    w_d, w_lin, w_rel, w_r = coef["log_dist"], coef["dist"], coef.get("log_dist_rel", 0.0), coef["r_hat"]
    # d logit / d d at 1 km for the median user, expressed in predicted-rating points
    slope_1km = w_d / (1 + 1.0) + w_lin + w_rel / (rho + 1.0)

    def popularity_needed(radius):
        """How many times more popular a place d km away must be to rank level with one at 0 km."""
        penalty = w_d * np.log1p(CURVE_KM) + w_lin * CURVE_KM + w_rel * np.log1p(CURVE_KM / radius)
        return np.exp(-penalty / coef["log_pop"]).tolist()

    out["model"] = {
        "chosen_features": best_fs, "lam": best_lam, "coef": coef, "intercept": b,
        "median_radius_km": rho, "stars_per_km_at_1km": float(-slope_1km / w_r),
        # one more km at 1 km is offset by a place this many times more popular
        "popularity_factor_per_km_at_1km": float(np.exp(-slope_1km / coef["log_pop"])),
        "standardized_coef": dict(zip(names, chosen.w.tolist())),
        "params": chosen.to_dict(), "columns": cols,
        "curve": {"km": CURVE_KM.tolist(),
                  "popularity_needed": {f"{r:g}": popularity_needed(r) for r in (0.5, round(rho, 1), 3.0)}},
        "radius_floor_km": float(feats.radius.min()), "fixed_tau": best_tau,
    }
    print(f"standardized coef {out['model']['standardized_coef']}", flush=True)
    print(f"1 more km at 1 km ~ {out['model']['stars_per_km_at_1km']:.2f} stars "
          f"~ x{out['model']['popularity_factor_per_km_at_1km']:.2f} popularity "
          f"(median radius {rho:.2f} km)", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "learned_distance.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
