"""Route ordering for the final places (RQ3) and the worked example (X1-X3).

For each sampled user: re-rank the hybrid candidates (exp decay, tau = 2 km),
keep the top N = 6, and order them from the anchor with every solver. Reports
the optimality gap of the heuristics (T3), the extra length of visiting in
recommendation order (T4), runtime against N (T2), and one example user.

Usage:  python experiments/route.py
Output: results/route.json
Targets (proposal 9.3): NN + 2-opt optimal for 82% of users (mean gap 0.9%);
recommendation order a median 27% longer than optimal; Held-Karp 1.7 ms at N = 8.
"""
import json
import time
from pathlib import Path

import numpy as np

from ranking import setup
from tasteroute.geo import distance_matrix
from tasteroute.rerank import rerank
from tasteroute.route import brute_force, held_karp, nearest_neighbor, nn_two_opt, path_length

ROOT = Path(__file__).resolve().parents[1]
N_PLACES = 6
TAU, DECAY = 2.0, "exp"


def route_matrix(anchor, items, places):
    """(N+1) x (N+1) distance matrix: the start point first, then the places."""
    lats = np.r_[anchor[0], items.lat.values[places]]
    lons = np.r_[anchor[1], items.lon.values[places]]
    return distance_matrix(lats, lons)


def timing(rng):
    """Mean runtime (ms) per solver over 10 random instances for N = 3..12."""
    out = []
    for n in range(3, 13):
        row = {"N": n}
        Ds = []
        for _ in range(10):
            pts = rng.uniform([39.9, -75.25], [40.05, -75.05], size=(n + 1, 2))
            Ds.append(distance_matrix(pts[:, 0], pts[:, 1]))
        for name, solver in [("held_karp", held_karp), ("nn_2opt", nn_two_opt), ("brute_force", brute_force)]:
            if name == "brute_force" and n > 9:
                continue  # 10! = 3.6M orders per instance; too slow to time
            t = time.perf_counter()
            for D in Ds:
                solver(D)
            row[name] = (time.perf_counter() - t) / len(Ds) * 1000
        out.append(row)
        print("timing", {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    return out


def main():
    """Run this step and write its output (see the module docstring)."""
    ctx = setup()
    items, anchors, dist, filt = ctx["items"], ctx["anchors"], ctx["dist"], ctx["filt"]
    hybrid = ctx["models"]["Hybrid"]
    n_train = np.bincount(ctx["train"].u.values, minlength=len(anchors))

    rows = []
    for u in ctx["users"]:
        scores = hybrid.score_all(u)
        cand = filt.top_m(u, scores)
        before = cand[:N_PLACES]  # rating order, no distance
        after = rerank(cand, scores[cand], dist(u, cand), TAU, DECAY, k=N_PLACES)
        D = route_matrix(anchors[u], items, after)
        opt = held_karp(D)[1]
        D_before = route_matrix(anchors[u], items, before)
        rows.append({
            "u": int(u),
            "optimal": opt,
            "rec_order": path_length(D, range(1, N_PLACES + 1)),
            "nn": nearest_neighbor(D)[1],
            "nn_2opt": nn_two_opt(D)[1],
            "before_rec_order": path_length(D_before, range(1, N_PLACES + 1)),
            "n_train": int(n_train[u]),
        })

    opt = np.array([r["optimal"] for r in rows])
    gap_nn = np.array([r["nn"] for r in rows]) / opt - 1
    gap_2opt = np.array([r["nn_2opt"] for r in rows]) / opt - 1
    rec_extra = np.array([r["rec_order"] for r in rows]) / opt - 1
    summary = {
        "nn_2opt_optimal_share": float(np.mean(gap_2opt < 1e-9)),
        "nn_2opt_mean_gap": float(gap_2opt.mean()),
        "nn_optimal_share": float(np.mean(gap_nn < 1e-9)),
        "nn_mean_gap": float(gap_nn.mean()),
        "rec_order_median_extra": float(np.median(rec_extra)),
        "saved_km_median": float(np.median([r["rec_order"] - r["optimal"] for r in rows])),
    }
    for k, v in summary.items():
        print(f"{k:<24} {v:.4f}", flush=True)

    # worked example: the user whose day shrinks the most from
    # "top 6 by rating, visited in rating order" to "re-ranked top 6, optimal order"
    ex = max(rows, key=lambda r: r["before_rec_order"] - r["optimal"])
    print(f"example user {ex['u']} ({ex['n_train']} training ratings): "
          f"{ex['before_rec_order']:.1f} km -> {ex['optimal']:.1f} km", flush=True)

    out = {"n_places": N_PLACES, "tau": TAU, "decay": DECAY, "summary": summary, "per_user": rows,
           "example": ex, "timing": timing(np.random.default_rng(42))}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "route.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
