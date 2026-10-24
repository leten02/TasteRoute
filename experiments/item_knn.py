"""Grid-search item-kNN (similarity x k x beta) on validation, then report test RMSE.

Usage:  python experiments/item_knn.py
Output: results/item_knn.json (grid for figure M1)
Target (proposal 9.5): cosine, k = 80, beta = 25, val 1.1400, test 1.142.
"""
import json
from pathlib import Path

from tasteroute.data import get_split, load_ratings, n_users_items
from tasteroute.metrics import evaluate
from tasteroute.models.item_knn import SIMILARITIES, ItemKNN

ROOT = Path(__file__).resolve().parents[1]
KS = [10, 20, 40, 80]
BETAS = [0, 25, 100]


def main():
    """Run this step and write its output (see the module docstring)."""
    r = load_ratings()
    n_users, n_items = n_users_items(r)
    train = get_split(r, "train")
    val = get_split(r, "val", warm_only=True)
    test = get_split(r, "test", warm_only=True)

    grid = []
    for sim in SIMILARITIES:
        for beta in BETAS:
            m = ItemKNN(n_users, n_items, sim, beta=beta).fit(train)
            for k in KS:  # k only changes prediction, so one fit serves all k
                m.k = k
                grid.append({"similarity": sim, "beta": beta, "k": k, "val_rmse": evaluate(m, val)["rmse"]})
                print(f"{sim:<9} beta={beta:<4} k={k:<3} val {grid[-1]['val_rmse']:.4f}", flush=True)

    best = min(grid, key=lambda g: g["val_rmse"])
    m = ItemKNN(n_users, n_items, best["similarity"], k=best["k"], beta=best["beta"]).fit(train)
    test_res = evaluate(m, test)
    print(f"best {best}  test RMSE {test_res['rmse']:.4f}")

    out = {"grid": grid, "best": best, "test": test_res}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "item_knn.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
