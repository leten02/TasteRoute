"""Load the preprocessed Philadelphia rating set.

All files come from data_analysis_preprocessing/preprocess.py and live in
data/processed/ (not committed, see data/README.md).

    ratings.pkl                 u, i, stars, date, split, cold_m
    items.pkl                   i, business_id, name, lat, lon, stars, review_count,
                                is_open, categories, price
    item_categories_tfidf.npz   items x categories, IDF-weighted (scipy sparse)
    categories.pkl              category name for each column of the matrix above

split is one of train / val / test / dropped. dropped holds the ratings removed
from cold-start users and is never used. cold_m is 0 for regular (warm) users
and 1, 3 or 5 for cold-start users, who keep only that many training ratings.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
HOLDOUT_SHARE = 0.15  # validation and test share per user, as in preprocess.py


def load_ratings():
    """Return all ratings as a DataFrame with integer user (u) and item (i) ids.

    pd.read_pickle(path) restores a DataFrame saved with DataFrame.to_pickle.
    """
    return pd.read_pickle(PROCESSED / "ratings.pkl")


def load_items():
    """Return one row per business, indexed 0..n_items-1 in column i."""
    return pd.read_pickle(PROCESSED / "items.pkl")


def load_item_features():
    """Return (X, names): the IDF-weighted item x category matrix and its column names.

    scipy.sparse.load_npz(path) loads a sparse matrix saved with save_npz. The
    result is a CSR matrix (compressed sparse row): only non-zero cells are
    stored, which matters here because 98% of the 7,693 x 188 cells are zero.
    """
    X = sparse.load_npz(PROCESSED / "item_categories_tfidf.npz").tocsr()
    names = pd.read_pickle(PROCESSED / "categories.pkl")
    return X, list(names)


def get_split(ratings, name, warm_only=False):
    """Return the ratings of one split ("train", "val" or "test").

    Args:
        ratings: DataFrame from load_ratings().
        name: split name.
        warm_only: keep only regular users (cold_m == 0). Model comparison uses
            warm users; cold-start users are evaluated separately by cold_m.
    """
    out = ratings[ratings.split == name]
    if warm_only:
        out = out[out.cold_m == 0]
    return out


def temporal_split(ratings):
    """Return a copy of ratings whose warm users are split by date instead of at random.

    preprocess.py shuffles each user's ratings before the 70 / 15 / 15 split, so
    a training rating can be newer than a test rating. Here every warm user
    (cold_m == 0) keeps the same split sizes, but the oldest ratings go to train,
    the next ones to val and the newest to test ("learn the past, predict the
    future"). Ties on date are broken by item id so the result is deterministic.
    Cold-start users keep their original split, so the training set they add
    is the same in both protocols.

    Sizes follow preprocess.py: n_val = n_test = max(1, round(0.15 * n)) and
    train gets the rest, which must be at least one rating.
    """
    out = ratings.copy()
    # work with row positions (_row), so a non-unique index cannot misplace a split
    rows = np.flatnonzero(out.cold_m.to_numpy() == 0)
    warm = out.iloc[rows][["u", "date", "i"]].assign(_row=rows).sort_values(["u", "date", "i"], kind="stable")
    g = warm.groupby("u")
    # transform("size") repeats each user's rating count on every row of that user;
    # cumcount() numbers a user's rows 0, 1, 2, ... in the (date) order above
    n = g.u.transform("size").to_numpy()
    pos = g.cumcount().to_numpy()
    # np.round rounds halves to even, the same rule as Python's round() in preprocess.py
    n_hold = np.maximum(1, np.round(HOLDOUT_SHARE * n)).astype(int)
    n_train = n - 2 * n_hold
    if (n_train < 1).any():
        raise ValueError("every warm user needs at least 3 ratings for a train / val / test split")
    col = out["split"].to_numpy(dtype=object).copy()
    col[warm._row.to_numpy()] = np.where(pos < n_train, "train", np.where(pos < n_train + n_hold, "val", "test"))
    out["split"] = col
    return out


def n_users_items(ratings):
    """Return (n_users, n_items) so every model sizes its arrays the same way.

    Sizes come from the full rating table, not the training split, so an item
    that appears only in val/test still has a slot (its learned bias stays 0).
    """
    return int(ratings.u.max()) + 1, int(ratings.i.max()) + 1
