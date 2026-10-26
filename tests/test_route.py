import itertools

import numpy as np

from tasteroute.geo import distance_matrix
from tasteroute.route import (brute_force, held_karp, held_karp_hours, nearest_neighbor, nn_two_opt, path_length,
                             schedule)


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


def _windows(n, seed, stay):
    """Random opening window (open, close) per place node 1..n, minutes from midnight."""
    rng = np.random.default_rng(seed)
    opens = rng.integers(600, 840, size=n + 1)
    closes = opens + rng.integers(30, 240, size=n + 1)

    def earliest(j, t):
        s = max(t, opens[j])
        return s if s + stay <= closes[j] else None
    return earliest


def test_hours_order_matches_brute_force():
    """Fewest closed visits, then earliest finish: same as trying every order."""
    stay, per_km = 20, 60 / 4.5
    for seed in range(60):
        n = 3 + seed % 5  # N = 3..7
        D = _random_D(n, seed)
        earliest = _windows(n, seed, stay)

        def cost(o):
            arrive, begin, closed = schedule(D, o, 600, per_km, stay, earliest)
            return sum(closed), begin[-1] + stay
        best = min(cost(o) for o in itertools.permutations(range(1, n + 1)))
        order, closed = held_karp_hours(D, 600, per_km, stay, earliest)
        assert sorted(order) == list(range(1, n + 1))
        assert closed == best[0] and np.isclose(cost(order)[1], best[1])


def test_hours_order_is_distance_order_when_all_open():
    for seed in range(30):
        D = _random_D(3 + seed % 6, seed)
        order, closed = held_karp_hours(D, 600, 13.3, 20, lambda j, t: t)
        assert closed == 0 and np.isclose(path_length(D, order), held_karp(D)[1])


def test_hours_order_avoids_a_closure():
    """Places on a line 0-1-2 with place 2 (far) closing at 10:30.

    The shortest order (near place first) reaches place 2 too late; going to
    place 2 first walks further but finds both open.
    """
    D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)  # km
    per_km, stay = 60 / 4.5, 30
    windows = {1: (0, 24 * 60), 2: (0, 10 * 60 + 30)}

    def earliest(j, t):
        s = max(t, windows[j][0])
        return s if s + stay <= windows[j][1] else None
    assert held_karp(D)[0] == [1, 2]
    assert sum(schedule(D, [1, 2], 9 * 60 + 30, per_km, stay, earliest)[2]) == 1
    order, closed = held_karp_hours(D, 9 * 60 + 30, per_km, stay, earliest)
    assert order == [2, 1] and closed == 0


def test_schedule_waits_for_opening():
    D = np.array([[0, 1], [1, 0]], dtype=float)
    arrive, begin, closed = schedule(D, [1], 600, 15, 30, lambda j, t: max(t, 660))
    assert arrive == [615] and begin == [660] and closed == [False]


def test_hours_order_small_cases():
    stay, per_km = 30, 15
    assert held_karp_hours(np.zeros((1, 1)), 600, per_km, stay, lambda j, t: t) == ([], 0)
    D1 = np.array([[0, 1], [1, 0]], dtype=float)
    assert held_karp_hours(D1, 600, per_km, stay, lambda j, t: t) == ([1], 0)        # open
    assert held_karp_hours(D1, 600, per_km, stay, lambda j, t: None) == ([1], 1)     # never fits
    # two places: place 2 closes at 11:10 (first: 10:30-11:00 fits; second: 11:00-11:30 does not)
    D2 = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)

    def earliest(j, t):
        return t if j == 1 or t + stay <= 670 else None
    assert held_karp_hours(D2, 600, per_km, stay, earliest) == ([2, 1], 0)
