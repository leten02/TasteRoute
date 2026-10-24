"""Philadelphia bakery tour: recommend n bakeries near a start point and map the route.

The talk opens with a Daejeon bakery tour as the motivating example; Yelp has
no Korean data, so the demo plans the same kind of day in Philadelphia.

Usage:
    python demo/bakery_tour.py                       # new visitor, starting at City Hall
    python demo/bakery_tour.py --user 1458           # personalized for a dataset user
    python demo/bakery_tour.py --lat 39.9496 --lon -75.1503 --n 5
Output: demo/bakery_tour.html (open in a browser)

The demo scores places with the CBF model (bias + category taste). It ties
the hybrid on RMSE (experiments/compare_models.py) and fits in seconds.
"""
import argparse
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from tasteroute.candidates import CandidateFilter
from tasteroute.data import get_split, load_item_features, load_items, load_ratings, n_users_items
from tasteroute.geo import user_anchors
from tasteroute.models.cbf import CBF
from tasteroute.pipeline import plan_day

HERE = Path(__file__).resolve().parent
CITY_HALL = (39.9526, -75.1636)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", type=int, help="dataset user id; omit for a new visitor")
    ap.add_argument("--lat", type=float, help="start latitude (default: user's anchor or City Hall)")
    ap.add_argument("--lon", type=float, help="start longitude")
    ap.add_argument("--n", type=int, default=6, help="number of bakeries (Held-Karp handles up to ~12)")
    ap.add_argument("--category", default="Bakeries", help="Yelp category to tour")
    args = ap.parse_args()

    r = load_ratings()
    items = load_items()
    n_users, n_items = n_users_items(r)
    X, _ = load_item_features()
    train = get_split(r, "train")
    model = CBF(n_users, n_items, X, lam=30).fit(train)
    # pd.concat([a, b]) stacks two DataFrames row-wise: places rated in train or
    # val are "seen" and never recommended again
    filt = CandidateFilter(items, train, seen=pd.concat([train, get_split(r, "val")]))

    if args.user is not None:
        pred = model.score_all(args.user)
        start = tuple(user_anchors(train, items, n_users)[args.user])
    else:
        # a new visitor has no user bias or taste vector: mu + b_i only
        pred = np.clip(model.bias.mu + model.bias.b_i, 1, 5)
        start = CITY_HALL
    if args.lat is not None and args.lon is not None:
        start = (args.lat, args.lon)

    # Series.str.contains(text, regex=False, na=False): True where the category
    # string contains `text` literally; places without categories count as False
    mask = items.categories.str.contains(args.category, regex=False, na=False).values
    plan = plan_day(pred, filt, items, start, user=args.user, n=args.n, mask=mask)
    if not plan["places"]:
        print(f"no open {args.category!r} place with enough ratings; check the --category spelling")
        return

    print(f"start {start[0]:.4f}, {start[1]:.4f}")
    for step, i in enumerate(plan["places"], start=1):
        row = items.iloc[i]
        print(f"{step}. {row['name']}  ({row['stars']} stars, {row['review_count']} reviews)")
    print(f"route {plan['route_km']:.1f} km  (recommendation order would be {plan['rec_order_km']:.1f} km)")

    # folium.Map(location, zoom_start) creates a Leaflet web map; Marker and
    # PolyLine add pins and the route line; m.save(path) writes a single HTML file.
    m = folium.Map(location=start, zoom_start=14, tiles="OpenStreetMap")
    folium.Marker(start, tooltip="Start", icon=folium.Icon(color="black", icon="home")).add_to(m)
    path = [start]
    for step, i in enumerate(plan["places"], start=1):
        row = items.iloc[i]
        loc = (row.lat, row.lon)
        path.append(loc)
        folium.Marker(
            loc,
            tooltip=f"{step}. {row['name']}",
            popup=f"{step}. {row['name']}<br>{row['stars']} stars, {row['review_count']} reviews",
            icon=folium.DivIcon(html=f'<div style="background:#2a78d6;color:#fff;border-radius:50%;'
                                     f'width:24px;height:24px;line-height:24px;text-align:center;'
                                     f'font:bold 13px sans-serif">{step}</div>'),
        ).add_to(m)
    folium.PolyLine(path, color="#2a78d6", weight=3).add_to(m)
    out = HERE / "bakery_tour.html"
    m.save(out)
    print(f"map: {out}")


if __name__ == "__main__":
    main()
