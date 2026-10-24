"""Tune the bias model's lambda on validation and report test RMSE for both baselines.

Usage:  python experiments/baseline.py
Output: results/baseline.json
Targets (proposal 9.5): global mean test RMSE 1.190; bias lam = 5, test 1.051.
"""
import json
from pathlib import Path

from tasteroute.data import get_split, load_ratings, n_users_items
from tasteroute.metrics import evaluate
from tasteroute.models.baseline import BiasModel, GlobalMean

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = [1, 5, 10, 25]


def main():
    r = load_ratings()
    n_users, n_items = n_users_items(r)
    train = get_split(r, "train")
    val = get_split(r, "val", warm_only=True)
    test = get_split(r, "test", warm_only=True)

    grid = []
    for lam in LAMBDAS:
        m = BiasModel(n_users, n_items, lam=lam).fit(train)
        grid.append({"lam": lam, "val_rmse": evaluate(m, val)["rmse"]})
        print(f"bias lam={lam:<3} val RMSE {grid[-1]['val_rmse']:.4f}")
    best = min(grid, key=lambda g: g["val_rmse"])

    out = {
        "global_mean": {"test": evaluate(GlobalMean(n_users, n_items).fit(train), test)},
        "bias": {
            "grid": grid,
            "best_lam": best["lam"],
            "test": evaluate(BiasModel(n_users, n_items, lam=best["lam"]).fit(train), test),
        },
    }
    print(f"global mean  test RMSE {out['global_mean']['test']['rmse']:.4f}")
    print(f"bias lam={best['lam']}  test RMSE {out['bias']['test']['rmse']:.4f}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "baseline.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
