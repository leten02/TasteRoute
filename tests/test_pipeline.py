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
