import numpy as np
import pandas as pd
import pytest

from tasteroute.data import temporal_split


def ratings(n, user=0, cold_m=0, shuffle_seed=0):
    """n ratings of one user on items 0..n-1, rated on days 1..n, rows shuffled."""
    df = pd.DataFrame({
        "u": user, "i": np.arange(n), "stars": 4,
        "date": pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n), unit="D"),
        "split": "train", "cold_m": cold_m,
    })
    return df.sample(frac=1, random_state=shuffle_seed)


def test_oldest_ratings_train_newest_test():
    r = ratings(10)
    t = temporal_split(r).sort_values("date")
    # n = 10: val = test = round(1.5) = 2 (half to even), train = 6
    assert list(t.split) == ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    assert t[t.split == "train"].date.max() < t[t.split == "test"].date.min()


def test_sizes_match_preprocess_rule_and_every_user_keeps_val_and_test():
    r = pd.concat([ratings(5, user=0), ratings(7, user=1), ratings(20, user=2)])
    t = temporal_split(r)
    for u, n in ((0, 5), (1, 7), (2, 20)):
        counts = t[t.u == u].split.value_counts()
        hold = max(1, round(0.15 * n))
        assert counts["val"] == hold and counts["test"] == hold and counts["train"] == n - 2 * hold


def test_cold_users_keep_their_split_and_input_is_not_changed():
    warm = ratings(6, user=0)
    cold = ratings(6, user=1, cold_m=3).assign(split=["train", "dropped", "val", "test", "dropped", "train"])
    r = pd.concat([warm, cold])
    before = r.copy()
    t = temporal_split(r)
    assert list(t[t.u == 1].split) == list(cold.split)
    pd.testing.assert_frame_equal(r, before)


def test_same_date_is_broken_by_item_id():
    r = ratings(5)
    r["date"] = pd.Timestamp("2020-01-01")
    t = temporal_split(r).sort_values("i")
    assert list(t.split) == ["train"] * 3 + ["val", "test"]


def test_too_few_ratings_raise():
    with pytest.raises(ValueError):
        temporal_split(ratings(2))
