"""Grid-search MF (f x lambda) on validation, then report test RMSE of the best setting.

Usage:  python experiments/mf.py
Output: results/mf.json (grid for figure M4, learning curve of the best run for M3)
Target (proposal 9.5): f = 10, lambda = 0.1, about 15 epochs, val 1.0438, test 1.052.
"""
import json
import time
from pathlib import Path

from tasteroute.data import get_split, load_ratings, n_users_items
from tasteroute.metrics import evaluate
from tasteroute.models.mf import MF

ROOT = Path(__file__).resolve().parents[1]
FS = [10, 20, 50, 100]
LAMBDAS = [0.01, 0.02, 0.05, 0.1]
ETA = 0.01


def main():
    r = load_ratings()
    n_users, n_items = n_users_items(r)
    train = get_split(r, "train")
    val = get_split(r, "val", warm_only=True)
    test = get_split(r, "test", warm_only=True)

    grid, best_model = [], None
    for f in FS:
        for lam in LAMBDAS:
            t = time.time()
            m = MF(n_users, n_items, f=f, eta=ETA, lam=lam).fit(train, val)
            val_rmse = min(h["val_rmse"] for h in m.history)
            grid.append({"f": f, "lam": lam, "val_rmse": val_rmse, "best_epoch": m.best_epoch})
            print(f"f={f:<3} lam={lam:<5} val {val_rmse:.4f} epoch {m.best_epoch:<3} ({time.time() - t:.0f}s)", flush=True)
            if best_model is None or val_rmse < min(g["val_rmse"] for g in grid[:-1]):
                best_model = m

    best = min(grid, key=lambda g: g["val_rmse"])
    test_res = evaluate(best_model, test)
    print(f"best f={best['f']} lam={best['lam']}  test RMSE {test_res['rmse']:.4f}")

    out = {"eta": ETA, "grid": grid, "best": best, "test": test_res, "curve": best_model.history}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "mf.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
