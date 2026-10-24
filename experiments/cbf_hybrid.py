"""Tune CBF (lambda_c) and the MF + CBF hybrid (alpha, kappa) on validation.

Usage:  python experiments/cbf_hybrid.py
Output: results/cbf_hybrid.json (CBF grid, alpha curve for M5, kappa curve for M6)
Targets (proposal 9.5): CBF lambda_c = 30, val 1.0426, test 1.051;
fixed alpha = 0.1, val 1.0441; adaptive kappa = 100, val 1.0426, test 1.051.
"""
import json
from pathlib import Path

import numpy as np

from tasteroute.data import get_split, load_item_features, load_ratings, n_users_items
from tasteroute.metrics import evaluate
from tasteroute.models.cbf import CBF
from tasteroute.models.hybrid import Hybrid
from tasteroute.models.mf import MF

ROOT = Path(__file__).resolve().parents[1]
CBF_LAMBDAS = [0.3, 1, 3, 10, 30]
ALPHAS = [round(float(a), 1) for a in np.linspace(0, 1, 11)]
KAPPAS = [0.5, 1, 2, 5, 10, 20, 30, 50, 100]
MF_BEST = {"f": 10, "lam": 0.1}  # from experiments/mf.py


def main():
    r = load_ratings()
    n_users, n_items = n_users_items(r)
    X, _ = load_item_features()
    train = get_split(r, "train")
    val = get_split(r, "val", warm_only=True)
    test = get_split(r, "test", warm_only=True)

    cbf_grid, cbf_models = [], {}
    for lam in CBF_LAMBDAS:
        cbf_models[lam] = CBF(n_users, n_items, X, lam=lam).fit(train)
        cbf_grid.append({"lam": lam, "val_rmse": evaluate(cbf_models[lam], val)["rmse"]})
        print(f"CBF lam={lam:<4} val {cbf_grid[-1]['val_rmse']:.4f}", flush=True)
    cbf_best = min(cbf_grid, key=lambda g: g["val_rmse"])
    cbf = cbf_models[cbf_best["lam"]]

    mf = MF(n_users, n_items, **MF_BEST).fit(train, val)
    print(f"MF val {min(h['val_rmse'] for h in mf.history):.4f}", flush=True)

    alpha_curve = []
    for a in ALPHAS:
        h = Hybrid(mf, cbf, alpha=a).fit(train)
        alpha_curve.append({"alpha": a, "val_rmse": evaluate(h, val)["rmse"]})
        print(f"hybrid alpha={a:<4} val {alpha_curve[-1]['val_rmse']:.4f}", flush=True)
    kappa_curve = []
    for kp in KAPPAS:
        h = Hybrid(mf, cbf, kappa=kp).fit(train)
        kappa_curve.append({"kappa": kp, "val_rmse": evaluate(h, val)["rmse"]})
        print(f"hybrid kappa={kp:<4} val {kappa_curve[-1]['val_rmse']:.4f}", flush=True)
    a_best = min(alpha_curve, key=lambda g: g["val_rmse"])
    k_best = min(kappa_curve, key=lambda g: g["val_rmse"])

    out = {
        "cbf": {"grid": cbf_grid, "best": cbf_best, "test": evaluate(cbf, test)},
        "hybrid_alpha": {"curve": alpha_curve, "best": a_best,
                         "test": evaluate(Hybrid(mf, cbf, alpha=a_best["alpha"]).fit(train), test)},
        "hybrid_kappa": {"curve": kappa_curve, "best": k_best,
                         "test": evaluate(Hybrid(mf, cbf, kappa=k_best["kappa"]).fit(train), test)},
    }
    for name in ("cbf", "hybrid_alpha", "hybrid_kappa"):
        print(f"{name:<13} best {out[name]['best']}  test RMSE {out[name]['test']['rmse']:.4f}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "cbf_hybrid.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
