"""RQ2 again: can richer place features help users with 1, 3 or 5 ratings? (issue #54)

Item features: IDF-weighted categories (as before) plus 42 Yelp attributes
(item_extras.py), the attribute block scaled by a weight w.
Models, all on the same bias baseline so only the content part differs:
    Bias          lambda tuned for cold users
    CBF           per-user ridge on categories           (lambda_c)
    CBF+attr      per-user ridge on categories+attributes (lambda_c, w)
    ContentKNN    content-similarity neighbors on categories+attributes (shrink, w)
Every setting is chosen on the cold users' validation ratings; test is scored
once, per cold group (cold_m = 1, 3, 5) and on warm users, with a paired t-test
of per-user squared error against the bias model.

Usage:  python experiments/cold_start.py    (needs python -m tasteroute.item_extras first)
Output: results/cold_start.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from tasteroute.data import get_split, load_item_features, load_ratings, n_users_items
from tasteroute.item_extras import attribute_matrix, combine_features, load_item_extras
from tasteroute.metrics import rmse
from tasteroute.models.baseline import BiasModel
from tasteroute.models.cbf import CBF
from tasteroute.models.content_knn import ContentKNN

ROOT = Path(__file__).resolve().parents[1]
BIAS_LAMBDAS = [2, 5, 10, 20, 50]
CBF_LAMBDAS = [10, 30, 100, 300, 1000, 3000]
WEIGHTS = [0.25, 0.5, 1, 2]
SHRINKS = [0.5, 1, 2, 5, 10, 20, 50, 200]


def per_user_mse(df, pred):
    """Mean squared error per user (only users present in df), as a dict u -> mse."""
    se = (df.stars.values - pred) ** 2
    s = df.assign(se=se).groupby("u").se.mean()
    return s


def main():
    """Run this step and write its output (see the module docstring)."""
    r = load_ratings()
    n_users, n_items = n_users_items(r)
    X, _ = load_item_features()
    A, attr_names = attribute_matrix(load_item_extras(), n_items)
    train = get_split(r, "train")
    val, test = get_split(r, "val"), get_split(r, "test")
    cold_val = val[val.cold_m > 0]
    print(f"cold validation ratings {len(cold_val):,}, attributes {len(attr_names)}", flush=True)

    def score(m):
        return rmse(cold_val.stars.values, m.predict(cold_val.u.values, cold_val.i.values))

    out = {"selection": {}, "test": {}, "paired_vs_bias": {}}
    grid = [{"lam": lam, "val": score(BiasModel(n_users, n_items, lam=lam).fit(train))} for lam in BIAS_LAMBDAS]
    out["selection"]["Bias"] = grid
    bias_lam = min(grid, key=lambda g: g["val"])["lam"]
    print(f"Bias: lambda {bias_lam} (cold val {min(g['val'] for g in grid):.4f})", flush=True)

    chosen = {"Bias": BiasModel(n_users, n_items, lam=bias_lam).fit(train)}

    def pick(name, configs, build):
        t = time.time()
        rows = []
        for cfg in configs:
            m = build(**cfg)
            rows.append({**cfg, "val": score(m)})
            if rows[-1]["val"] == min(x["val"] for x in rows):
                chosen[name] = m
        out["selection"][name] = rows
        b = min(rows, key=lambda x: x["val"])
        print(f"{name}: {({k: v for k, v in b.items() if k != 'val'})} cold val {b['val']:.4f} "
              f"({time.time() - t:.0f}s)", flush=True)

    pick("CBF", [{"lam": lam} for lam in CBF_LAMBDAS],
         lambda lam: CBF(n_users, n_items, X, lam=lam, bias_lam=bias_lam).fit(train))
    pick("CBF+attr", [{"lam": lam, "w": w} for w in WEIGHTS for lam in CBF_LAMBDAS],
         lambda lam, w: CBF(n_users, n_items, combine_features(X, A, w), lam=lam, bias_lam=bias_lam).fit(train))
    pick("ContentKNN", [{"shrink": s, "w": w} for w in WEIGHTS for s in SHRINKS],
         lambda shrink, w: ContentKNN(n_users, n_items, combine_features(X, A, w), shrink=shrink,
                                      bias_lam=bias_lam).fit(train))

    groups = {"cold m=1": test[test.cold_m == 1], "cold m=3": test[test.cold_m == 3],
              "cold m=5": test[test.cold_m == 5], "warm": test[test.cold_m == 0]}
    bias_err = {g: per_user_mse(df, chosen["Bias"].predict(df.u.values, df.i.values)) for g, df in groups.items()}
    for name, m in chosen.items():
        out["test"][name], out["paired_vs_bias"][name] = {}, {}
        for g, df in groups.items():
            pred = m.predict(df.u.values, df.i.values)
            out["test"][name][g] = rmse(df.stars.values, pred)
            if name != "Bias":
                e = per_user_mse(df, pred)
                # stats.ttest_rel: paired t-test on the same users' mean squared error
                p = float(stats.ttest_rel(e.values, bias_err[g].loc[e.index].values).pvalue)
                out["paired_vs_bias"][name][g] = p if np.isfinite(p) else None
        cells = "  ".join(f"{g} {v:.4f}" for g, v in out["test"][name].items())
        print(f"test {name:<11} {cells}", flush=True)
    for name, ps in out["paired_vs_bias"].items():
        if ps:
            print(f"p vs bias {name:<11} " + "  ".join(f"{g} {p:.3g}" for g, p in ps.items()), flush=True)
    out["n_test"] = {g: len(df) for g, df in groups.items()}
    out["n_users"] = {g: int(df.u.nunique()) for g, df in groups.items()}

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "cold_start.json").write_text(json.dumps(out, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
