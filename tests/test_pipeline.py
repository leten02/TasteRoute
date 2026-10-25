import numpy as np
import pandas as pd

from tasteroute.candidates import CandidateFilter
from tasteroute.pipeline import plan_day


def test_plan_day_picks_near_places_and_orders_them():
    # four equally popular, equally rated places on a north-south line; one far away
    items = pd.DataFrame({
        "i": range(5), "is_open": 1,
        "lat": [40.001, 40.003, 40.002, 40.004, 40.5], "lon": [-75.0] * 5,
    })
    train = pd.DataFrame({"u": np.repeat(range(10), 5), "i": np.tile(range(5), 10)})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    plan = plan_day(np.full(5, 4.0), filt, items, start=(40.0, -75.0), n=4)
    assert plan["places"] == [0, 2, 1, 3]  # nearest first, far place 4 left out
    assert plan["route_km"] <= plan["rec_order_km"] + 1e-9


def test_plan_day_respects_mask():
    items = pd.DataFrame({"i": range(3), "is_open": 1, "lat": [40.0, 40.01, 40.02], "lon": [-75.0] * 3})
    train = pd.DataFrame({"u": [0, 0, 0], "i": [0, 1, 2]})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    plan = plan_day(np.full(3, 4.0), filt, items, (40.0, -75.0), n=2, mask=np.array([False, True, True]))
    assert sorted(plan["places"]) == [1, 2]


def test_plan_day_with_no_or_few_candidates():
    items = pd.DataFrame({"i": range(3), "is_open": 1, "lat": [40.0, 40.01, 40.02], "lon": [-75.0] * 3})
    train = pd.DataFrame({"u": [0, 0, 0], "i": [0, 1, 2]})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    empty = plan_day(np.full(3, 4.0), filt, items, (40.0, -75.0), n=2, mask=np.zeros(3, dtype=bool))
    assert empty["places"] == [] and empty["route_km"] == 0
    few = plan_day(np.full(3, 4.0), filt, items, (40.0, -75.0), n=5)
    assert sorted(few["places"]) == [0, 1, 2]


def test_plan_day_rejects_out_of_range_n():
    import pytest

    items = pd.DataFrame({"i": range(3), "is_open": 1, "lat": [40.0, 40.01, 40.02], "lon": [-75.0] * 3})
    train = pd.DataFrame({"u": [0, 0, 0], "i": [0, 1, 2]})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    for n in (0, -1, 30):
        with pytest.raises(ValueError):
            plan_day(np.full(3, 4.0), filt, items, (40.0, -75.0), n=n)


def test_plan_day_with_learned_visit_model():
    from tasteroute.geo import user_anchors
    from tasteroute.models.visit import LogisticRegression, VisitFeatures

    items = pd.DataFrame({
        "i": range(4), "is_open": 1,
        "lat": [40.001, 40.002, 40.003, 40.3], "lon": [-75.0] * 4,
    })
    train = pd.DataFrame({"u": np.repeat(range(5), 4), "i": np.tile(range(4), 5)})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    feats = VisitFeatures(items, train, user_anchors(train, items, 5), filt.popularity)
    # only distance matters, and nearer is better
    model = LogisticRegression.from_dict({"lam": 0, "mean": [0, 0, 0], "std": [1, 1, 1], "w": [-3.0, 0, 0], "b": 0})
    plan = plan_day(np.full(4, 4.0), filt, items, (40.0, -75.0), n=3, visit=(model, feats, [0, 3, 4]))
    assert plan["ranked"] == [0, 1, 2] and 3 not in plan["places"]
    assert np.isclose(sum(plan["legs_km"]), plan["route_km"])
    assert len(plan["rec_path"]) == 4


def test_plan_days_splits_by_neighborhood():
    from tasteroute.pipeline import plan_days

    # two clusters of four places, far apart
    lat = [40.0, 40.001, 40.002, 40.003, 40.1, 40.101, 40.102, 40.103]
    items = pd.DataFrame({"i": range(8), "is_open": 1, "lat": lat, "lon": [-75.0] * 8})
    train = pd.DataFrame({"u": np.repeat(range(5), 8), "i": np.tile(range(8), 5)})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    plan = plan_days(np.full(8, 4.0), filt, items, (40.05, -75.0), days=2, per_day=4, tau=100)
    days = [set(d["places"]) for d in plan["days"]]
    assert len(days) == 2 and {frozenset(d) for d in days} == {frozenset(range(4)), frozenset(range(4, 8))}


def test_plan_day_diversity_needs_features_and_changes_picks():
    import pytest
    from scipy import sparse

    items = pd.DataFrame({"i": range(3), "is_open": 1, "lat": [40.0, 40.0001, 40.0002], "lon": [-75.0] * 3})
    train = pd.DataFrame({"u": [0, 0, 0, 1, 1, 2], "i": [0, 1, 2, 0, 1, 0]})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    F = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    plain = plan_day(np.full(3, 4.0), filt, items, (40.0, -75.0), n=2)
    varied = plan_day(np.full(3, 4.0), filt, items, (40.0, -75.0), n=2, diversity=0.7, features=F)
    assert set(plain["places"]) == {0, 1} and set(varied["places"]) == {0, 2}
    with pytest.raises(ValueError):
        plan_day(np.full(3, 4.0), filt, items, (40.0, -75.0), n=2, diversity=0.5)


def test_plan_days_allows_more_than_max_places_in_total():
    from tasteroute.pipeline import MAX_PLACES, plan_days

    k = 3 * 6
    items = pd.DataFrame({"i": range(k), "is_open": 1, "lat": 40.0 + np.arange(k) * 0.002, "lon": [-75.0] * k})
    train = pd.DataFrame({"u": np.repeat(range(3), k), "i": np.tile(range(k), 3)})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    plan = plan_days(np.full(k, 4.0), filt, items, (40.0, -75.0), days=3, per_day=6)
    assert k > MAX_PLACES and [len(d["places"]) for d in plan["days"]] == [6, 6, 6]


def test_plan_days_spreads_few_places_and_fetches_streets_once():
    from tasteroute.pipeline import plan_days
    from tasteroute.roads import RoadNetwork

    lat = [40.0, 40.001, 40.002, 40.1, 40.101, 40.102, 40.2]
    items = pd.DataFrame({"i": range(7), "is_open": 1, "lat": lat, "lon": [-75.0] * 7})
    train = pd.DataFrame({"u": np.repeat(range(3), 7), "i": np.tile(range(7), 3)})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    calls = []
    grid = RoadNetwork(np.array(lat), np.full(7, -75.0), np.arange(6), np.arange(1, 7), np.full(6, 1.0))

    def streets(lats, lons):
        calls.append(len(lats))
        return grid

    plan = plan_days(np.full(7, 4.0), filt, items, (40.0, -75.0), days=3, per_day=6, tau=100, roads=streets)
    assert len(calls) == 1 and calls[0] == 8  # one network for start + all 7 places
    assert sorted(len(d["places"]) for d in plan["days"]) == [1, 3, 3]  # cap ceil(7 / 3) = 3, all days used


def test_learned_model_with_diversity_and_no_columns():
    from scipy import sparse

    from tasteroute.geo import user_anchors
    from tasteroute.models.visit import LogisticRegression, VisitFeatures

    items = pd.DataFrame({"i": range(4), "is_open": 1, "lat": [40.0, 40.001, 40.002, 40.3], "lon": [-75.0] * 4})
    train = pd.DataFrame({"u": np.repeat(range(5), 4), "i": np.tile(range(4), 5)})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    feats = VisitFeatures(items, train, user_anchors(train, items, 5), filt.popularity)
    model = LogisticRegression.from_dict({"lam": 0, "mean": [0] * 5, "std": [1] * 5, "w": [-3.0, 0, 0, 0, 0], "b": 0})
    F = sparse.csr_matrix(np.eye(4))
    plan = plan_day(np.full(4, 4.0), filt, items, (40.0, -75.0), n=2, visit=(model, feats, None),
                    diversity=0.3, features=F)
    assert len(plan["places"]) == 2 and 3 not in plan["places"]
