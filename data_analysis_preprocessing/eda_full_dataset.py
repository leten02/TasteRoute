"""First look at the full Yelp dataset, used to choose the city for the project.

Prints business and review statistics (rating distribution, sparsity, reviews per
user and per business, duplicates) and runs 5-core / 10-core filtering for the
15 cities with the most reviews.

Usage:  python data_analysis_preprocessing/eda_full_dataset.py
Input:  data/raw/yelp_academic_dataset_{business,review}.json
Output: data_analysis_preprocessing/results/city_summary.csv,
        data/processed/reviews_slim.pkl (not committed)
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "raw"
OUT = HERE / "results"
SLIM = HERE.parent / "data" / "processed"
t0 = time.time()


def log(*a):
    print(*a, flush=True)


# ---------- business ----------
rows = []
with open(f"{DATA}/yelp_academic_dataset_business.json", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        rows.append((d["business_id"], d["name"], d["city"], d["state"], d["latitude"], d["longitude"],
                     d["stars"], d["review_count"], d["is_open"], d["categories"]))
biz = pd.DataFrame(rows, columns=["business_id", "name", "city", "state", "lat", "lon", "stars",
                                  "review_count", "is_open", "categories"])
log("== BUSINESS")
log("n_business", len(biz))
log("null categories", biz.categories.isna().sum(), "null lat", biz.lat.isna().sum(), "null lon", biz.lon.isna().sum())
log("is_open share", round(biz.is_open.mean(), 3))
biz["cat_list"] = biz.categories.fillna("").map(lambda s: [c.strip() for c in s.split(",") if c.strip()])
biz["is_rest"] = biz.cat_list.map(lambda l: "Restaurants" in l or "Food" in l)
log("restaurants/food share", round(biz.is_rest.mean(), 3))
cc = Counter(c for l in biz.cat_list for c in l)
log("n unique categories", len(cc))
log("cats per business mean/median", round(biz.cat_list.map(len).mean(), 2), biz.cat_list.map(len).median())
log("top 25 categories", cc.most_common(25))
log("states", biz.state.value_counts().head(20).to_dict())

# ---------- reviews (slim) ----------
uids, bids, stars, years = [], [], [], []
with open(f"{DATA}/yelp_academic_dataset_review.json", encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        uids.append(d["user_id"]); bids.append(d["business_id"])
        stars.append(d["stars"]); years.append(int(d["date"][:4]))
        if i % 1_000_000 == 0:
            log(f"  reviews read {i:,} ({time.time()-t0:.0f}s)")
rev = pd.DataFrame({"user_id": pd.Categorical(uids), "business_id": pd.Categorical(bids),
                    "stars": np.array(stars, dtype=np.int8), "year": np.array(years, dtype=np.int16)})
del uids, bids, stars, years
SLIM.mkdir(parents=True, exist_ok=True)
rev.to_pickle(SLIM / "reviews_slim.pkl")

log("== REVIEWS")
nu, nb = rev.user_id.nunique(), rev.business_id.nunique()
log("n_reviews", len(rev), "n_users", nu, "n_business", nb)
log("sparsity", 1 - len(rev) / (nu * nb))
log("stars dist", rev.stars.value_counts(normalize=True).sort_index().round(3).to_dict(), "mean", round(rev.stars.mean(), 3))
log("year dist", rev.year.value_counts().sort_index().to_dict())
dup = rev.duplicated(["user_id", "business_id"]).sum()
log("duplicate user-business pairs", dup)
upc = rev.user_id.value_counts()
bpc = rev.business_id.value_counts()
for name, s in [("per-user", upc), ("per-business", bpc)]:
    log(name, "quantiles", s.quantile([.25, .5, .75, .9, .99]).to_dict(), "mean", round(s.mean(), 2), "max", s.max())
log("users with 1 review share", round((upc == 1).mean(), 3), ">=5", round((upc >= 5).mean(), 3), ">=10", round((upc >= 10).mean(), 3))

# ---------- per city ----------
rev = rev.merge(biz[["business_id", "city", "state", "is_rest"]].assign(
    business_id=lambda x: x.business_id.astype(str)), left_on=rev.business_id.astype(str), right_on="business_id",
    how="left", suffixes=("", "_b"))
rev["metro"] = rev.city.str.strip().str.title() + ", " + rev.state


def kcore(df, k):
    df = df.drop_duplicates(["user_id", "business_id"])
    while True:
        u = df.user_id.map(df.user_id.value_counts())
        b = df.business_id.map(df.business_id.value_counts())
        keep = (u >= k) & (b >= k)
        if keep.all():
            return df
        df = df[keep]


log("== CITIES (top 15 by reviews)")
top = rev.metro.value_counts().head(15)
res = []
for m in top.index:
    d = rev[rev.metro == m][["user_id", "business_id", "stars", "is_rest"]].copy()
    d["user_id"] = d.user_id.astype(str); d["business_id"] = d.business_id.astype(str)
    row = {"metro": m, "reviews": len(d), "users": d.user_id.nunique(), "biz": d.business_id.nunique()}
    for k in (5, 10):
        c = kcore(d, k)
        row[f"k{k}_reviews"] = len(c); row[f"k{k}_users"] = c.user_id.nunique(); row[f"k{k}_biz"] = c.business_id.nunique()
        row[f"k{k}_density%"] = round(100 * len(c) / max(1, row[f"k{k}_users"] * row[f"k{k}_biz"]), 3)
    c = kcore(d[d.is_rest == True], 5)
    row["rest_k5_reviews"] = len(c); row["rest_k5_users"] = c.user_id.nunique(); row["rest_k5_biz"] = c.business_id.nunique()
    res.append(row)
    log(row)
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(res).to_csv(OUT / "city_summary.csv", index=False)
log(f"done in {time.time()-t0:.0f}s")
