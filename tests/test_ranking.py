import math

import numpy as np
import pandas as pd

from tasteroute.candidates import CandidateFilter
from tasteroute.geo import distance_matrix, haversine, user_anchors
from tasteroute.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k
from tasteroute.rerank import distance_weight, rerank

CITY_HALL = (39.9526, -75.1636)
TIMES_SQUARE = (40.7580, -73.9855)


def test_ranking_metrics_by_hand():
    ranked, rel = [10, 11, 12], {11}
    assert math.isclose(precision_at_k(ranked, rel, 3), 1 / 3)
    assert recall_at_k(ranked, rel, 3) == 1.0
    assert hit_rate_at_k(ranked, rel, 3) == 1.0
    assert math.isclose(ndcg_at_k(ranked, rel, 3), 1 / math.log2(3))  # ~0.631
    assert ndcg_at_k([11, 10, 12], rel, 3) == 1.0
    assert ndcg_at_k([10, 12], rel, 2) == 0.0


def test_haversine_known_distances():
    assert haversine(*CITY_HALL, *CITY_HALL) == 0
    d = haversine(*CITY_HALL, *TIMES_SQUARE)
    assert 130 < d < 138  # about 134 km
    assert math.isclose(d, haversine(*TIMES_SQUARE, *CITY_HALL))
    D = distance_matrix([CITY_HALL[0], TIMES_SQUARE[0]], [CITY_HALL[1], TIMES_SQUARE[1]])
    assert D.shape == (2, 2) and D[0, 0] == 0 and math.isclose(D[0, 1], d)


def test_anchor_is_median_not_mean():
    items = pd.DataFrame({"i": [0, 1, 2], "lat": [0.0, 1.0, 50.0], "lon": [0.0, 1.0, 50.0]})
    train = pd.DataFrame({"u": [0, 0, 0], "i": [0, 1, 2]})
    a = user_anchors(train, items, n_users=2)
    assert a[0, 0] == 1.0  # median ignores the far trip; mean would be ~17
    assert np.isnan(a[1, 0])


def test_distance_weights():
    assert distance_weight(0, 2) == 1.0
    assert math.isclose(distance_weight(2 * math.log(2), 2), 0.5)  # exp halves every tau ln 2
    assert list(distance_weight([1, 3], 2, "cutoff")) == [1.0, 0.0]


def test_rerank_prefers_near_when_ratings_are_close():
    out = rerank([7, 8], pred=[4.7, 4.6], dist=[15.0, 1.0], tau=2, k=2)
    assert list(out) == [8, 7]
    assert list(rerank([7, 8], [4.7, 4.6], [15.0, 1.0], tau=1e6, k=2)) == [7, 8]  # huge tau = rating order


def test_candidates_exclude_seen_closed_and_rare():
    items = pd.DataFrame({"i": [0, 1, 2, 3], "is_open": [1, 1, 0, 1]})
    train = pd.DataFrame({"u": [0] * 3 + [1] * 3, "i": [0, 1, 2, 0, 1, 2]})
    f = CandidateFilter(items, train, seen=train[train.u == 0], min_support=2)
    top = f.top_m(0, scores=np.array([5.0, 4.0, 3.0, 2.0]), m=3)
    assert len(top) == 0  # 0,1 seen, 2 closed, 3 has no support
    top = f.top_m(1, scores=np.array([5.0, 4.0, 3.0, 2.0]), m=3)
    assert list(top) == [0, 1]


def test_distance_weight_rejects_non_positive_tau():
    import pytest

    from tasteroute.rerank import distance_weight

    for tau in (0, -1):
        with pytest.raises(ValueError):
            distance_weight([0.0, 1.0], tau)


def test_ranking_metrics_reject_bad_k_and_count_duplicates_once():
    import pytest

    from tasteroute.metrics import ndcg_at_k, precision_at_k

    with pytest.raises(ValueError):
        precision_at_k([1, 2], {1}, 0)
    assert ndcg_at_k([11, 11], {11}, 2) == 1.0  # not 1.63
