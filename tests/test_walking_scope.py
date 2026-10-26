"""Regression tests for issue #66; use synthetic coordinates, no Yelp or OSM downloads."""
import importlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from tasteroute.geo import haversine
from tasteroute.roads import CENTER_CITY_BOX, KM_PER_DEG_LAT, RoadNetwork

# walking.py uses the same sibling import of ranking.py when run as a script.
with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "experiments"), *sys.path]):
    walking = importlib.import_module("walking")

NORTH, SOUTH, EAST, WEST = CENTER_CITY_BOX
MID_LAT, MID_LON = (NORTH + SOUTH) / 2, (EAST + WEST) / 2


def network(box=CENTER_CITY_BOX):
    """A network with explicit extents; no streets are needed for scope checks."""
    north, south, east, west = box
    return RoadNetwork([north, south], [east, west], [], [], [])


def wide_network():
    return network((NORTH + 0.1, SOUTH - 0.1, EAST + 0.1, WEST - 0.1))


class WalkingScopeTests(unittest.TestCase):
    def test_larger_cache_does_not_admit_places_outside_any_city_edge(self):
        lat = [MID_LAT, NORTH + 0.01, SOUTH - 0.01, MID_LAT, MID_LAT]
        lon = [MID_LON, MID_LON, MID_LON, EAST + 0.01, WEST - 0.01]
        np.testing.assert_array_equal(
            walking.inside_scope(wide_network(), lat, lon), [True, False, False, False, False])

    def test_larger_cache_keeps_the_same_item_and_user_scope(self):
        lat = [MID_LAT, NORTH - 0.001, SOUTH - 0.01, MID_LAT]
        lon = [MID_LON, MID_LON, MID_LON, EAST - 0.001]
        for margin in (0.0, walking.EDGE_KM):
            with self.subTest(margin=margin):
                np.testing.assert_array_equal(
                    walking.inside_scope(network(), lat, lon, margin),
                    walking.inside_scope(wide_network(), lat, lon, margin))

    def test_user_margin_is_measured_from_all_fixed_city_edges(self):
        # Check both sides of the 0.5 km threshold, even with a much wider cache.
        dlat = walking.EDGE_KM / KM_PER_DEG_LAT
        dlon = dlat / np.cos(np.radians(MID_LAT))
        lat = [NORTH - dlat * 0.9, NORTH - dlat * 1.1,
               SOUTH + dlat * 0.9, SOUTH + dlat * 1.1,
               MID_LAT, MID_LAT, MID_LAT, MID_LAT]
        lon = [MID_LON] * 4 + [EAST - dlon * 0.9, EAST - dlon * 1.1,
                               WEST + dlon * 0.9, WEST + dlon * 1.1]
        np.testing.assert_array_equal(
            walking.inside_scope(wide_network(), lat, lon, walking.EDGE_KM),
            [False, True, False, True, False, True, False, True])

    def test_items_on_city_boundary_are_included_without_user_margin(self):
        np.testing.assert_array_equal(
            walking.inside_scope(wide_network(), [NORTH, SOUTH, MID_LAT, MID_LAT],
                                 [MID_LON, MID_LON, EAST, WEST]),
            [True, True, True, True])

    def test_network_coverage_is_still_required(self):
        small = network((MID_LAT + 0.005, MID_LAT - 0.005, MID_LON + 0.005, MID_LON - 0.005))
        np.testing.assert_array_equal(
            walking.inside_scope(small, [MID_LAT, NORTH - 0.001], [MID_LON, MID_LON]),
            [True, False])

    def test_empty_and_nonfinite_coordinates_are_excluded(self):
        self.assertEqual(walking.inside_scope(wide_network(), [], []).shape, (0,))
        np.testing.assert_array_equal(
            walking.inside_scope(wide_network(), [np.nan, np.inf, MID_LAT],
                                 [MID_LON, MID_LON, np.nan]),
            [False, False, False])

    def test_main_filters_validation_test_users_and_candidate_places(self):
        # Six places in the city, one north of it; all seven pass CandidateFilter.
        item_lat = np.array([MID_LAT + x for x in (-0.004, -0.002, 0.0, 0.002, 0.004, 0.006)]
                            + [NORTH + 0.02])
        item_lon = np.array([MID_LON + x for x in (-0.004, 0.002, -0.002, 0.004, 0.0, 0.006)]
                            + [MID_LON])
        items = pd.DataFrame({"i": np.arange(7), "lat": item_lat, "lon": item_lon, "is_open": 1})
        # User 1 is outside the city; user 2 is within 0.5 km of its northern edge.
        anchors = np.array([[MID_LAT, MID_LON], [NORTH + 0.02, MID_LON],
                            [NORTH - 0.001, MID_LON], [MID_LAT - 0.002, MID_LON + 0.001]])
        train = pd.DataFrame({"u": np.repeat(np.arange(10, 30), 7), "i": np.tile(np.arange(7), 20)})
        val = pd.DataFrame({"u": np.arange(4), "i": [0, 1, 2, 3], "stars": 5,
                            "split": "val", "cold_m": 0})
        model = SimpleNamespace(score_all=Mock(return_value=np.arange(7, dtype=float) + 1))
        ctx = {"items": items, "train": train, "models": {"Hybrid": model}, "anchors": anchors,
               "users": np.arange(4), "relevant": [{0}, {1}, {2}, {3}],
               "filt": walking.CandidateFilter(items, train, seen=train)}
        # A tiny connected street graph covering a much larger box than Center City.
        lat = np.r_[item_lat, anchors[:, 0], NORTH + 0.1, SOUTH - 0.1]
        lon = np.r_[item_lon, anchors[:, 1], EAST + 0.1, WEST - 0.1]
        u, v = np.where(~np.eye(len(lat), dtype=bool))
        net = RoadNetwork(lat, lon, u, v, haversine(lat[u], lon[u], lat[v], lon[v]))
        distances = Mock(wraps=net.distances_from)
        net.distances_from = distances
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(walking, "ROOT", Path(tmp)), \
                patch.object(walking, "setup", return_value=ctx), \
                patch.object(walking, "load_ratings", return_value=val), \
                patch.object(walking.RoadNetwork, "download", return_value=net), \
                patch.object(walking.stats, "ttest_rel", return_value=SimpleNamespace(pvalue=1.0)), \
                redirect_stdout(io.StringIO()):
            walking.main()
            result = json.loads((Path(tmp) / "results" / "walking.json").read_text())
        self.assertEqual(result["n_val_users"], 2)
        self.assertEqual(result["n_test_users"], 2)
        self.assertEqual({r["u"] for r in result["rq3"]["per_user"]}, {0, 3})
        self.assertEqual({call.args[0] for call in model.score_all.call_args_list}, {0, 3})
        self.assertEqual(distances.call_count, 4)  # two validation and two test pools
        for call in distances.call_args_list:
            self.assertEqual(len(call.args[2]), 6)
            self.assertTrue(np.all(call.args[2] <= NORTH))


if __name__ == "__main__":
    unittest.main()
