"""Place attributes and opening hours from the raw Yelp business file.

The preprocessing step keeps only categories and price. Yelp also lists
attributes (outdoor seating, Wi-Fi, noise level, ambience, good for brunch,
...) and weekly opening hours, which help in two places:
    - content-based filtering for users with few ratings (RQ2): more item
      features than categories alone
    - day planning: only visit places that are open at the time you arrive

Usage:  python -m tasteroute.item_extras    (reads data/raw/, writes data/processed/item_extras.pkl)
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "yelp_academic_dataset_business.json"
OUT = ROOT / "data" / "processed" / "item_extras.pkl"
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

BOOLEAN = ["RestaurantsTakeOut", "RestaurantsDelivery", "OutdoorSeating", "RestaurantsReservations", "BikeParking",
           "HasTV", "RestaurantsGoodForGroups", "GoodForKids", "Caters", "RestaurantsTableService",
           "WheelchairAccessible", "HappyHour", "DogsAllowed", "DriveThru", "BYOB"]
# attribute -> values that become their own 0/1 feature (Yelp stores them as "u'free'" etc.)
CHOICES = {
    "Alcohol": ["full_bar", "beer_and_wine"],
    "WiFi": ["free"],
    "NoiseLevel": ["quiet", "loud", "very_loud"],
    "RestaurantsAttire": ["dressy", "formal"],
}
# attribute holding a dict of sub-flags -> sub-flags kept
NESTED = {
    "Ambience": ["romantic", "intimate", "classy", "hipster", "divey", "touristy", "trendy", "upscale", "casual"],
    "GoodForMeal": ["dessert", "latenight", "lunch", "dinner", "brunch", "breakfast"],
    "BusinessParking": ["garage", "street", "lot", "valet"],
}


def _value(raw):
    """Yelp attribute strings like "True", "u'free'", "{'romantic': False}" as Python values.

    ast.literal_eval(s) parses a Python literal (bool, str, dict, None) without
    running code; anything it cannot parse is returned as the plain string.
    """
    try:
        return ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return raw


def attribute_features(attrs):
    """0/1 features for one business (dict from the raw file, or None)."""
    attrs = attrs or {}
    out = {f"attr_{a}": float(_value(attrs.get(a)) is True) for a in BOOLEAN}
    for a, choices in CHOICES.items():
        v = _value(attrs.get(a))
        for c in choices:
            out[f"attr_{a}_{c}"] = float(isinstance(v, str) and v.strip("u'\"") == c)
    for a, keys in NESTED.items():
        v = _value(attrs.get(a))
        v = v if isinstance(v, dict) else {}
        for k in keys:
            out[f"attr_{a}_{k}"] = float(v.get(k) is True)
    return out


def parse_hours(hours):
    """{day: (open_minute, close_minute)} from Yelp's {"Monday": "8:0-22:30", ...}.

    Minutes count from that day's midnight; a close at or before the opening time
    means after midnight, so 24 * 60 is added (e.g. "17:0-2:0" -> (1020, 1560)).
    "0:0-0:0" is Yelp's notation for open all day.
    """
    out = {}
    for day, span in (hours or {}).items():
        try:
            a, b = span.split("-")
            ah, am = (int(x) for x in a.split(":"))
            bh, bm = (int(x) for x in b.split(":"))
        except ValueError:
            continue
        start, end = ah * 60 + am, bh * 60 + bm
        if end <= start:
            end += 24 * 60
        out[day] = (start, end)
    return out


def is_open_at(hours, day, minute):
    """True if a place with parsed hours is open on `day` at `minute` after midnight.

    Also counts the previous day's opening that runs past midnight. Places with
    no hours listed are treated as open (unknown, not closed).
    """
    if not hours:
        return True
    k = DAYS.index(day)
    start, end = hours.get(day, (None, None))
    if start is not None and start <= minute < end:
        return True
    prev = hours.get(DAYS[k - 1])
    return prev is not None and minute + 24 * 60 < prev[1]


def build(items, raw=RAW):
    """DataFrame indexed by item id: attribute columns plus a `hours` column."""
    want = dict(zip(items.business_id, items.i))
    rows = {}
    with open(raw) as f:
        for line in f:
            b = json.loads(line)
            i = want.get(b["business_id"])
            if i is None:
                continue
            rows[i] = {**attribute_features(b.get("attributes")), "hours": parse_hours(b.get("hours"))}
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return df.reindex(np.sort(items.i.values))


def attribute_matrix(extras, n_items):
    """IDF-weighted item x attribute matrix (scipy sparse CSR), rows by item id.

    Same weighting as the category features: a 0/1 attribute j gets weight
    log(N / df_j), so rare attributes (romantic, dressy) count more than common
    ones (takes out). Items without a raw record get an all-zero row.
    """
    from scipy import sparse

    cols = [c for c in extras.columns if c.startswith("attr_")]
    A = np.zeros((n_items, len(cols)))
    A[extras.index.values] = extras[cols].fillna(0).values
    df = A.sum(axis=0)
    idf = np.log(n_items / np.maximum(df, 1))
    return sparse.csr_matrix(A * idf), cols


def combine_features(X_categories, A_attributes, weight):
    """[categories | weight * attributes] as one sparse matrix for CBF.

    scipy.sparse.hstack([A, B]) puts matrices side by side (same rows); weight
    sets how much the attribute block counts against the category block.
    """
    from scipy import sparse

    return sparse.hstack([X_categories, weight * A_attributes]).tocsr()


def load_item_extras(path=OUT):
    """The DataFrame written by build() (run python -m tasteroute.item_extras first)."""
    return pd.read_pickle(path)


if __name__ == "__main__":
    from .data import load_items

    extras = build(load_items())
    extras.to_pickle(OUT)
    attr = [c for c in extras.columns if c.startswith("attr_")]
    print(f"{len(extras):,} places, {len(attr)} attribute features, "
          f"{int((extras.hours.map(len) > 0).sum()):,} with opening hours -> {OUT}")
