import numpy as np
import pandas as pd

from tasteroute.candidates import CandidateFilter
from tasteroute.geo import haversine
from tasteroute.pipeline import plan_day
from tasteroute.roads import RoadNetwork

STEP = 0.001  # degrees between grid streets (~0.1 km)


def _grid(n=3, skip=()):
    """n x n street grid, two-way edges, minus the undirected edges in `skip`."""
    lat = np.repeat(np.arange(n) * STEP + 40.0, n)
    lon = np.tile(np.arange(n) * STEP - 75.0, n)
    u, v = [], []
    for r in range(n):
        for c in range(n):
            k = r * n + c
            for k2 in ([k + 1] if c + 1 < n else []) + ([k + n] if r + 1 < n else []):
                if (k, k2) not in skip:
                    u += [k, k2]
                    v += [k2, k]
    u, v = np.array(u), np.array(v)
    return RoadNetwork(lat, lon, u, v, haversine(lat[u], lon[u], lat[v], lon[v])), lat, lon


def test_walking_distance_goes_around_the_block():
    net, lat, lon = _grid()
    r = net.routes(lat[[0, 4]], lon[[0, 4]])  # opposite corners of one block
    straight = haversine(lat[0], lon[0], lat[4], lon[4])
    two_sides = haversine(lat[0], lon[0], lat[1], lon[1]) + haversine(lat[1], lon[1], lat[4], lon[4])
    assert np.isclose(r.D[0, 1], two_sides) and r.D[0, 1] > straight
    path = r.path(0, 1)
    assert len(path) == 5  # point, 3 street nodes (corner, corner, corner), point


def test_detour_when_a_street_is_missing():
    net, lat, lon = _grid(skip={(0, 1)})
    r = net.routes(lat[[0, 1]], lon[[0, 1]])
    across = haversine(lat[0], lon[0], lat[1], lon[1])
    up = haversine(lat[0], lon[0], lat[3], lon[3])
    assert np.isclose(r.D[0, 1], up + across + up)  # up, across, down


def test_points_snap_to_nearest_node():
    net, lat, lon = _grid()
    r = net.routes([lat[0] + 0.0001, lat[8]], [lon[0], lon[8]])
    assert list(r.nodes) == [0, 8]
    assert r.D[0, 1] > r.D[1, 0] - 1e-9 and np.isclose(r.D[0, 1], r.D[1, 0])


def test_unreachable_pair_falls_back_to_straight_line():
    lat, lon = np.array([40.0, 40.001, 40.1]), np.array([-75.0, -75.0, -75.0])
    net = RoadNetwork(lat, lon, [0, 1], [1, 0], [0.11, 0.11])  # node 2 has no streets
    r = net.routes(lat[[0, 2]], lon[[0, 2]])
    assert not r.connected[0, 1]
    assert np.isclose(r.D[0, 1], haversine(lat[0], lon[0], lat[2], lon[2]))
    assert len(r.path(0, 1)) == 2


def test_network_without_streets():
    net = RoadNetwork([40.0, 40.001], [-75.0, -75.0], [], [], [])
    r = net.routes([40.0, 40.001], [-75.0, -75.0])
    assert not r.connected[0, 1] and np.isclose(r.D[0, 1], haversine(40.0, -75.0, 40.001, -75.0))


def test_tour_path_with_no_places_keeps_the_start():
    net, lat, lon = _grid()
    assert net.routes(lat[[0]], lon[[0]]).tour_path([]) == [(lat[0], lon[0])]


def test_parallel_edges_keep_the_shortest():
    net = RoadNetwork([40.0, 40.001], [-75.0, -75.0], [0, 0, 1], [1, 1, 0], [0.5, 0.2, 0.2])
    assert np.isclose(net.graph[0, 1], 0.2)


def test_plan_day_routes_on_streets():
    net, lat, lon = _grid(n=4)
    items = pd.DataFrame({"i": range(3), "is_open": 1, "lat": lat[[5, 10, 15]], "lon": lon[[5, 10, 15]]})
    train = pd.DataFrame({"u": [0, 0, 0], "i": [0, 1, 2]})
    filt = CandidateFilter(items, train, seen=train.iloc[:0], min_support=1)
    plan = plan_day(np.full(3, 4.0), filt, items, (lat[0], lon[0]), n=3, roads=net)
    assert plan["places"] == [0, 1, 2]
    assert plan["route_km"] >= plan["straight_km"]
    assert plan["path"][0] == (lat[0], lon[0]) and len(plan["path"]) > 4

    flat = plan_day(np.full(3, 4.0), filt, items, (lat[0], lon[0]), n=3)
    assert np.isclose(flat["route_km"], flat["straight_km"]) and len(flat["path"]) == 4
