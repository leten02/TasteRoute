import numpy as np

from tasteroute.geo import distance_matrix
from tasteroute.route import brute_force, held_karp, nearest_neighbor, nn_two_opt, path_length


def _random_D(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform([39.9, -75.25], [40.05, -75.05], size=(n + 1, 2))
    return distance_matrix(pts[:, 0], pts[:, 1])


def test_held_karp_matches_brute_force():
    for seed in range(100):
        D = _random_D(3 + seed % 6, seed)  # N = 3..8
        hk_order, hk_len = held_karp(D)
        _, bf_len = brute_force(D)
        assert np.isclose(hk_len, bf_len)
        assert np.isclose(path_length(D, hk_order), hk_len)
        assert sorted(hk_order) == list(range(1, len(D)))


def test_small_cases():
    D = _random_D(1, 0)
    assert held_karp(D)[0] == [1] and brute_force(D)[0] == [1]
    D = _random_D(2, 1)
    assert np.isclose(held_karp(D)[1], brute_force(D)[1])


def test_points_on_a_line_are_visited_in_order():
    D = distance_matrix([40.0, 40.01, 40.03, 40.02], [-75.0] * 4)
    assert held_karp(D)[0] == [1, 3, 2]


def test_heuristics_are_valid_and_never_beat_optimum():
    for seed in range(50):
        D = _random_D(6, seed)
        opt = held_karp(D)[1]
        for solver in (nearest_neighbor, nn_two_opt):
            order, length = solver(D)
            assert sorted(order) == list(range(1, 7))
            assert length >= opt - 1e-9
        assert nn_two_opt(D)[1] <= nearest_neighbor(D)[1] + 1e-9
