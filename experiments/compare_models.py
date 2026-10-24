"""Compare all rating models at their tuned settings (figures V1, V5).

Warm test users: RMSE and MAE, a 95% bootstrap interval over users, and a paired
t-test of per-user squared error against the bias model.
Cold-start users (cold_m = 1, 3, 5 training ratings): RMSE per group (RQ2).

Usage:  python experiments/compare_models.py
Output: results/model_compare.json
Settings come from experiments/{baseline,mf,item_knn,cbf_hybrid}.py.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from tasteroute.data import get_split, load_item_features, load_ratings, n_users_items
from tasteroute.metrics import mae, rmse
from tasteroute.models.baseline import BiasModel, GlobalMean
from tasteroute.models.cbf import CBF
from tasteroute.models.hybrid import Hybrid
from tasteroute.models.item_knn import ItemKNN
from tasteroute.models.mf import MF

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 1000
SEED = 42


def per_user_sse(df, pred):
    """Per-user sum of squared errors and rating counts, indexed by user id."""
    n = int(df.u.max()) + 1
    se = (df.stars.values - pred) ** 2
    return np.bincount(df.u.values, weights=se, minlength=n), np.bincount(df.u.values, minlength=n)


def bootstrap_rmse(sse, cnt, rng):
    """95% interval of RMSE when resampling users (not ratings) with replacement."""
    users = np.flatnonzero(cnt)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.choice(users, size=len(users), replace=True)
        vals[b] = np.sqrt(sse[s].sum() / cnt[s].sum())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    r = load_ratings()
    n_users, n_items = n_users_items(r)
    X, _ = load_item_features()
    train = get_split(r, "train")
    val = get_split(r, "val", warm_only=True)
    test = get_split(r, "test")
    warm = test[test.cold_m == 0]

    print("fitting models", flush=True)
    bias = BiasModel(n_users, n_items, lam=5).fit(train)
    mf = MF(n_users, n_items, f=10, lam=0.1).fit(train, val)
    cbf = CBF(n_users, n_items, X, lam=30).fit(train)
    models = {
        "Global mean": GlobalMean(n_users, n_items).fit(train),
        "Bias": bias,
        "Item-kNN": ItemKNN(n_users, n_items, "cosine", k=80, beta=0).fit(train),
        "MF": mf,
        "CBF": cbf,
        "Hybrid": Hybrid(mf, cbf, kappa=100).fit(train),
    }

    rng = np.random.default_rng(SEED)
    # bias errors first, so every other model (Global mean included) gets a p-value
    sse_bias, _ = per_user_sse(warm, bias.predict(warm.u.values, warm.i.values))
    out = {"warm": {}, "cold": {}}
    for name, m in models.items():
        pred = m.predict(warm.u.values, warm.i.values)
        sse, cnt = per_user_sse(warm, pred)
        users = np.flatnonzero(cnt)
        res = {
            "rmse": rmse(warm.stars.values, pred),
            "mae": mae(warm.stars.values, pred),
            "rmse_ci95": bootstrap_rmse(sse, cnt, rng),
        }
        if name != "Bias":
            # stats.ttest_rel(a, b): paired t-test on matched samples a[k], b[k]
            # (here the same user's mean squared error under two models).
            # Undefined p (e.g. a single user) is stored as null, not NaN.
            p = float(stats.ttest_rel(sse[users] / cnt[users], sse_bias[users] / cnt[users]).pvalue)
            res["vs_bias_p"] = p if np.isfinite(p) else None
        out["warm"][name] = res

        out["cold"][name] = {}
        for m_ in (1, 3, 5):
            g = test[test.cold_m == m_]
            out["cold"][name][m_] = rmse(g.stars.values, m.predict(g.u.values, g.i.values))

        ci = res["rmse_ci95"]
        cold = "  ".join(f"m={k}: {v:.4f}" for k, v in out["cold"][name].items())
        p = f"  p vs bias {res['vs_bias_p']:.3g}" if res.get("vs_bias_p") is not None else ""
        print(f"{name:<12} RMSE {res['rmse']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]  MAE {res['mae']:.4f}{p}  | cold {cold}",
              flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    # allow_nan=False: fail loudly instead of writing NaN, which is not valid JSON
    (ROOT / "results" / "model_compare.json").write_text(json.dumps(out, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
