"""Preprocess the Yelp Open Dataset into a Philadelphia restaurant rating set.

Steps: radius scope -> Restaurants/Food filter -> dedup (latest rating) ->
iterative k-core -> category TF-IDF matrix -> per-user split + cold-start group.
Also writes the statistics behind the EDA and preprocessing figures.

Usage: python data_analysis_preprocessing/preprocess.py
Input:  data/raw/yelp_academic_dataset_{business,review}.json
Output: data/processed/ (not committed), data_analysis_preprocessing/results/eda_stats.json
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
RES = HERE / "results"

CENTER = (39.9526, -75.1636)  # Philadelphia City Hall
RADIUS_KM = 15.0
K_CORE = 5
MIN_CAT_ITEMS = 10
GENERIC_CATS = {"Restaurants", "Food"}
SEED = 42
COLD_SHARE = 0.10
COLD_MIN_RATINGS = 10
COLD_M = (1, 3, 5)

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def kcore(df, k):
    while True:
        u = df.user_id.map(df.user_id.value_counts())
        b = df.business_id.map(df.business_id.value_counts())
        keep = (u >= k) & (b >= k)
        if keep.all():
            return df
        df = df[keep]


def hist(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    return [{"lo": float(lo), "hi": float(hi), "n": int(n)} for lo, hi, n in zip(edges[:-1], edges[1:], counts)]


# ---------- businesses ----------
rows = []
with open(DATA / "yelp_academic_dataset_business.json", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        rows.append((d["business_id"], d["name"], d["city"], d["state"], d["latitude"], d["longitude"],
                     d["stars"], d["review_count"], d["is_open"], d["categories"] or "",
                     (d.get("attributes") or {}).get("RestaurantsPriceRange2")))
biz = pd.DataFrame(rows, columns=["business_id", "name", "city", "state", "lat", "lon", "stars",
                                  "review_count", "is_open", "categories", "price"])
biz["dist_center"] = haversine(CENTER[0], CENTER[1], biz.lat.values, biz.lon.values)
biz["cat_list"] = biz.categories.map(lambda s: [c.strip() for c in s.split(",") if c.strip()])
biz["is_rest"] = biz.cat_list.map(lambda l: bool(GENERIC_CATS & set(l)))
in_radius = set(biz.business_id[biz.dist_center <= RADIUS_KM])
in_scope = set(biz.business_id[(biz.dist_center <= RADIUS_KM) & biz.is_rest])
city_field = set(biz.business_id[biz.city.str.strip().str.lower() == "philadelphia"])
log("businesses", len(biz), "in radius", len(in_radius), "restaurants in radius", len(in_scope))

# ---------- reviews ----------
n_raw = 0
n_radius = 0
n_city_field = 0
recs = []
with open(DATA / "yelp_academic_dataset_review.json", encoding="utf-8") as f:
    for line in f:
        n_raw += 1
        d = json.loads(line)
        b = d["business_id"]
        if b in city_field:
            n_city_field += 1
        if b in in_radius:
            n_radius += 1
            if b in in_scope:
                recs.append((d["user_id"], b, int(d["stars"]), d["date"]))
rev = pd.DataFrame(recs, columns=["user_id", "business_id", "stars", "date"])
rev["date"] = pd.to_datetime(rev.date)
log("reviews raw", n_raw, "radius", n_radius, "restaurants", len(rev))

rev = rev.sort_values("date").drop_duplicates(["user_id", "business_id"], keep="last")
n_dedup = len(rev)
pre = rev.copy()  # scope after dedup, before k-core
log("after dedup", n_dedup)

# k-core shrinkage curve (E13)
shrink = []
for k in range(1, 21):
    c = kcore(pre, k)
    shrink.append({"k": k, "ratings": len(c), "users": int(c.user_id.nunique()), "items": int(c.business_id.nunique())})
log("shrink curve done")

post = kcore(pre, K_CORE).copy()
log("after k-core", len(post), post.user_id.nunique(), post.business_id.nunique())

# ---------- index ----------
users = np.sort(post.user_id.unique())
items_ids = np.sort(post.business_id.unique())
uidx = {u: n for n, u in enumerate(users)}
iidx = {b: n for n, b in enumerate(items_ids)}
post["u"] = post.user_id.map(uidx)
post["i"] = post.business_id.map(iidx)
items = biz.set_index("business_id").loc[items_ids].reset_index()
items["i"] = np.arange(len(items))

# ---------- categories ----------
cat_lists = items.cat_list.map(lambda l: [c for c in l if c not in GENERIC_CATS])
cat_count = pd.Series([c for l in cat_lists for c in l]).value_counts()
kept_cats = sorted(cat_count[cat_count >= MIN_CAT_ITEMS].index)
cidx = {c: n for n, c in enumerate(kept_cats)}
r_idx, c_idx = [], []
for i, l in enumerate(cat_lists):
    for c in l:
        if c in cidx:
            r_idx.append(i)
            c_idx.append(cidx[c])
X = sparse.csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(len(items), len(kept_cats)))
df_c = np.asarray(X.sum(axis=0)).ravel()
idf = np.log(len(items) / (1 + df_c))
Xw = X.multiply(idf).tocsr()
norms = np.sqrt(np.asarray(Xw.multiply(Xw).sum(axis=1)).ravel())
norms[norms == 0] = 1.0
X_tfidf = sparse.diags(1 / norms) @ Xw
log("categories kept", len(kept_cats))

# ---------- split ----------
rng = np.random.default_rng(SEED)
post = post.sample(frac=1.0, random_state=SEED).sort_values("u", kind="stable").reset_index(drop=True)
split = np.empty(len(post), dtype=object)
for u, idx in post.groupby("u").indices.items():
    n = len(idx)
    n_val = max(1, round(0.15 * n))
    n_test = max(1, round(0.15 * n))
    split[idx[: n - n_val - n_test]] = "train"
    split[idx[n - n_val - n_test: n - n_test]] = "val"
    split[idx[n - n_test:]] = "test"
post["split"] = split

counts = post.groupby("u").size()
eligible = counts[counts >= COLD_MIN_RATINGS].index.values
cold_users = rng.choice(eligible, size=int(COLD_SHARE * len(eligible)), replace=False)
cold_m = {u: COLD_M[n % len(COLD_M)] for n, u in enumerate(rng.permutation(cold_users))}
post["cold_m"] = post.u.map(cold_m).fillna(0).astype(int)
for u, m in cold_m.items():
    tr = post.index[(post.u == u) & (post.split == "train")]
    post.loc[tr[m:], "split"] = "dropped"
log("cold-start users", len(cold_m))

# ---------- save ----------
OUT.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
post[["u", "i", "stars", "date", "split", "cold_m"]].to_pickle(OUT / "ratings.pkl")
items[["i", "business_id", "name", "lat", "lon", "stars", "review_count", "is_open", "categories", "price"]].to_pickle(OUT / "items.pkl")
pd.Series(users).to_pickle(OUT / "users.pkl")
sparse.save_npz(OUT / "item_categories_tfidf.npz", X_tfidf.tocsr())
with open(OUT / "categories.pkl", "wb") as f:
    pickle.dump(kept_cats, f)
log("saved to", OUT)

# ---------- figure statistics ----------
stats = {"center": CENTER, "radius_km": RADIUS_KM, "k": K_CORE}
stats["stages"] = [
    {"stage": "Raw reviews", "ratings": n_raw},
    {"stage": "Philadelphia city field", "ratings": n_city_field},
    {"stage": f"Within {RADIUS_KM:.0f} km of City Hall", "ratings": n_radius},
    {"stage": "Restaurants and Food", "ratings": int(len(recs))},
    {"stage": "Deduplicated", "ratings": n_dedup},
    {"stage": f"{K_CORE}-core", "ratings": int(len(post))},
]
stats["pre"] = {"ratings": n_dedup, "users": int(pre.user_id.nunique()), "items": int(pre.business_id.nunique())}
stats["post"] = {"ratings": int(len(post)), "users": int(len(users)), "items": int(len(items)),
                 "density": len(post) / (len(users) * len(items))}
stats["shrink"] = shrink

# E3 CCDF
xs = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
nu = pre.user_id.value_counts()
nb = pre.business_id.value_counts()
stats["ccdf"] = ([{"who": "users", "x": x, "share": float((nu >= x).mean())} for x in xs if (nu >= x).any()]
                 + [{"who": "businesses", "x": x, "share": float((nb >= x).mean())} for x in xs if (nb >= x).any()])

# E4 Lorenz + Gini on business popularity (pre)
v = np.sort(nb.values)
cum = np.cumsum(v) / v.sum()
qs = np.linspace(0, 1, 51)
stats["lorenz"] = [{"pop": float(q), "share": float(cum[max(0, int(q * len(v)) - 1)]) if q > 0 else 0.0} for q in qs]
n = len(v)
stats["gini"] = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * v) / (n * v.sum()))
stats["top10_share"] = float(v[int(0.9 * n):].sum() / v.sum())

# E6 spatial grid (post items)
ir = post.groupby("i").size()
items["n_ratings"] = items.i.map(ir)
step = 0.01
items["gx"] = np.floor(items.lon / step).astype(int)
items["gy"] = np.floor(items.lat / step).astype(int)
grid = items.groupby(["gx", "gy"]).agg(n_items=("i", "size"), n_ratings=("n_ratings", "sum")).reset_index()
stats["grid"] = [{"lon": round((r.gx + 0.5) * step, 4), "lat": round((r.gy + 0.5) * step, 4),
                  "items": int(r.n_items), "ratings": int(r.n_ratings)} for r in grid.itertuples()]
stats["grid_step"] = step

# E7 pairwise distance
samp = items.sample(n=min(3000, len(items)), random_state=SEED)
la, lo = samp.lat.values, samp.lon.values
iu = np.triu_indices(len(samp), k=1)
pd_km = haversine(la[iu[0]], lo[iu[0]], la[iu[1]], lo[iu[1]])
stats["pair_dist"] = hist(pd_km, np.arange(0, 31, 1.0))
stats["pair_dist_median"] = float(np.median(pd_km))

# E8 user activity radius
m = post.merge(items[["i", "lat", "lon"]], on="i")
g = m.groupby("u")
anchor = g[["lat", "lon"]].median()
m = m.join(anchor, on="u", rsuffix="_a")
m["d_anchor"] = haversine(m.lat_a.values, m.lon_a.values, m.lat.values, m.lon.values)
cent = g[["lat", "lon"]].mean()
m = m.join(cent, on="u", rsuffix="_c")
m["d_cent"] = haversine(m.lat_c.values, m.lon_c.values, m.lat.values, m.lon.values)
radius = m.groupby("u").d_cent.mean()
stats["user_radius"] = hist(radius.clip(upper=19.99), np.arange(0, 20.5, 0.5))
stats["user_radius_q"] = {str(q): float(radius.quantile(q)) for q in (0.25, 0.5, 0.75, 0.9)}
stats["anchor_dist_q"] = {str(q): float(m.d_anchor.quantile(q)) for q in (0.25, 0.5, 0.75, 0.9)}

# E10 category co-occurrence (Jaccard, top 15)
top_c = list(cat_count.index[:15])
sets = {c: set(np.where(cat_lists.map(lambda l, c=c: c in l))[0]) for c in top_c}
stats["cooc"] = [{"a": a, "b": b, "jaccard": len(sets[a] & sets[b]) / len(sets[a] | sets[b])}
                 for a in top_c for b in top_c]
stats["cat_top"] = [{"cat": c, "n": int(cat_count[c])} for c in top_c]

# E11 user / item mean rating
edges = np.arange(1.0, 5.01, 0.25)
stats["user_mean"] = hist(post.groupby("u").stars.mean().clip(upper=4.999), edges)
stats["item_mean"] = hist(post.groupby("i").stars.mean().clip(upper=4.999), edges)

# E12 popularity vs quality
im = post.groupby("i").stars.agg(["mean", "size"])
bins = [5, 10, 20, 50, 100, 200, 500, 10_000]
im["bin"] = pd.cut(im["size"], bins=bins, right=False)
pq = im.groupby("bin", observed=True)["mean"].describe()
stats["pop_quality"] = [{"bin": f"{int(b.left)}-{int(b.right) - 1}" if b.right < 10_000 else f"{int(b.left)}+",
                         "n": int(r["count"]), "mean": float(r["mean"]), "p25": float(r["25%"]), "p75": float(r["75%"])}
                        for b, r in pq.iterrows()]

# P2 rating distribution before / after
stats["rating_pre_post"] = [{"stars": s, "pre": float((pre.stars == s).mean()), "post": float((post.stars == s).mean())}
                            for s in range(1, 6)]

# P3 centered ratings (train)
tr = post[post.split == "train"]
centered = tr.stars - tr.u.map(tr.groupby("u").stars.mean())
stats["centered"] = hist(centered.clip(-3.999, 3.999), np.arange(-4, 4.01, 0.5))
stats["user_std_median"] = float(tr.groupby("u").stars.std().median())

# P4 split sizes
sc = post.split.value_counts()
stats["split"] = {k: int(sc.get(k, 0)) for k in ("train", "val", "test", "dropped")}
stats["cold"] = {str(mm): int(sum(1 for v_ in cold_m.values() if v_ == mm)) for mm in COLD_M}

# P5 categories per business after pruning
cpb = np.asarray(X.sum(axis=1)).ravel()
stats["cats_per_item"] = [{"n_cats": int(k_), "items": int((cpb == k_).sum())} for k_ in range(0, int(cpb.max()) + 1)]
stats["n_kept_cats"] = len(kept_cats)
stats["n_all_cats_scope"] = int(len(cat_count))

with open(RES / "eda_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=1)
log("stats saved to", RES / "eda_stats.json")
