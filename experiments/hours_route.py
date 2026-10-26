"""Visiting order with opening hours vs distance only (issue #68).

For random day plans (test user, weekday, start time, category) the dashboard's
candidate set is used: places open for the whole stay at some arrival time
during the day (open_mask over the day's window). The top 6 are then ordered
twice from the user's anchor, on straight-line km:

    distance    Held-Karp, shortest path (what the app did before)
    hours       route.held_karp_hours: fewest visits that find the place
                closed, then the earliest finish (waiting for openings counts)

Both orders are walked with the same timeline (4.5 km/h, 25 min per place) and
compared on closed visits, km, wait and finish time (paired, per plan).

Usage:  python experiments/hours_route.py
Output: results/hours_route.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tasteroute.candidates import CandidateFilter
from tasteroute.data import get_split, load_item_features, load_items, load_ratings, n_users_items
from tasteroute.geo import user_anchors
from tasteroute.item_extras import DAYS, load_item_extras, open_mask
from tasteroute.models.cbf import CBF
from tasteroute.pipeline import plan_day

ROOT = Path(__file__).resolve().parents[1]
SEED, N_PLANS, N_PLACES, STAY = 42, 400, 6, 25
STARTS = [8 * 60, 10 * 60, 12 * 60, 14 * 60, 17 * 60, 19 * 60]
# all places plus categories whose hours differ a lot (morning, evening, daytime)
CATEGORIES = ["", "", "Breakfast & Brunch", "Bars", "Coffee & Tea", "Bakeries", "Desserts", "Pizza"]


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    r, items = load_ratings(), load_items()
    n_users, n_items = n_users_items(r)
    X, _ = load_item_features()
    train = get_split(r, "train")
    cbf = CBF(n_users, n_items, X, lam=30).fit(train)
    filt = CandidateFilter(items, train, seen=pd.concat([train, get_split(r, "val")]))
    anchors = user_anchors(train, items, n_users)
    extras = load_item_extras()
    hours = extras.hours

    def hours_of(i):
        h = hours.get(i)
        return h if isinstance(h, dict) and h else None

    users = rng.choice(np.unique(get_split(r, "test").u.values), N_PLANS, replace=False)
    rows = []
    for u in users:
        day, start_min = DAYS[rng.integers(7)], int(rng.choice(STARTS))
        category = CATEGORIES[rng.integers(len(CATEGORIES))]
        mask = open_mask(extras, n_items, day, start_min, STAY, N_PLACES * (STAY + 15))
        if category:
            mask &= items.categories.str.contains(category, regex=False, na=False).values
        start = tuple(anchors[u])
        plan = plan_day(cbf.score_all(u), filt, items, start, int(u), N_PLACES, mask,
                        when=(day, start_min, STAY, hours_of))
        if len(plan["places"]) < N_PLACES:
            continue
        rows.append({
            "user": int(u), "day": day, "start": start_min, "category": category or "all",
            "closed_distance": plan["distance_order_closed"], "closed_hours": int(sum(plan["closed"])),
            "km_distance": plan["distance_order_km"], "km_hours": plan["route_km"],
            "wait_distance": plan["distance_order_wait_min"], "wait_hours": plan["wait_min"],
            "end_distance": plan["distance_order_end_min"], "end_hours": plan["end_min"],
        })
    df = pd.DataFrame(rows)
    changed = (df.km_hours - df.km_distance).abs() > 1e-9

    def paired(a, b):
        diff = df[b] - df[a]
        p = float(stats.wilcoxon(df[a], df[b]).pvalue) if (diff != 0).any() else 1.0
        return {"distance": float(df[a].mean()), "hours": float(df[b].mean()), "mean_change": float(diff.mean()),
                "wilcoxon_p": p}

    out = {
        "setup": {"plans": len(df), "places_per_plan": N_PLACES, "stay_min": STAY, "distance": "straight km",
                  "seed": SEED, "starts": STARTS, "categories": sorted(set(CATEGORIES))},
        "plans_with_closure": {"distance": int((df.closed_distance > 0).sum()), "hours": int((df.closed_hours > 0).sum())},
        "closed_visits_total": {"distance": int(df.closed_distance.sum()), "hours": int(df.closed_hours.sum())},
        "order_changed": int(changed.sum()),
        "closed_per_plan": paired("closed_distance", "closed_hours"),
        "km": paired("km_distance", "km_hours"),
        "wait_min": paired("wait_distance", "wait_hours"),
        "finish_min": paired("end_distance", "end_hours"),
        # among plans whose order changed: what the change cost and bought
        "when_changed": {
            "plans": int(changed.sum()),
            "closures_avoided": int((df.closed_distance - df.closed_hours)[changed].sum()),
            "extra_km_mean": float((df.km_hours - df.km_distance)[changed].mean()) if changed.any() else 0.0,
            "extra_km_max": float((df.km_hours - df.km_distance)[changed].max()) if changed.any() else 0.0,
            "less_wait_min_mean": float((df.wait_distance - df.wait_hours)[changed].mean()) if changed.any() else 0.0,
        },
        "by_category": {c: {"plans": int(len(g)), "closed_distance": int(g.closed_distance.sum()),
                            "closed_hours": int(g.closed_hours.sum())}
                        for c, g in df.groupby("category")},
        "seconds": round(time.time() - t0, 1),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "hours_route.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
