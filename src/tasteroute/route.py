"""Route ordering: visit every recommended place once, starting from the anchor.

Open-path traveling salesman problem on a distance matrix D of size (N+1) x (N+1):
node 0 is the start (user anchor or hotel), nodes 1..N are the places, and the
path ends at the last place (no return to the start).

Every solver returns (order, length): order lists the place nodes 1..N in
visiting order (node 0 is implied first), length is the total distance.
Proposal section 7.
"""
import itertools

import numpy as np


def path_length(D, order):
    """Length of the open path 0 -> order[0] -> ... -> order[-1]."""
    nodes = [0, *order]
    return float(sum(D[a, b] for a, b in zip(nodes[:-1], nodes[1:])))


def brute_force(D):
    """Exact: try all N! orders. itertools.permutations(seq) yields every ordering."""
    places = range(1, len(D))
    best = min(itertools.permutations(places), key=lambda o: path_length(D, o))
    return list(best), path_length(D, best)


def held_karp(D):
    """Exact: Held-Karp dynamic programming in O(N^2 2^N).

    C[S][j] = shortest path that starts at 0, visits exactly the place set S
    (a bitmask over places 0..N-1, i.e. nodes 1..N) and ends at place j in S.

        C[{j}][j] = D[0][j]
        C[S][j]   = min_{k in S, k != j} C[S - {j}][k] + D[k][j]

    The answer is min_j C[all][j]; parent[S][j] stores the best k so the order
    can be read back from the end.
    """
    n = len(D) - 1
    if n == 0:
        return [], 0.0
    full = (1 << n) - 1
    C = np.full((1 << n, n), np.inf)
    parent = np.full((1 << n, n), -1, dtype=int)
    for j in range(n):
        C[1 << j, j] = D[0, j + 1]
    for S in range(1, full + 1):
        for j in range(n):
            if not S & (1 << j) or S == 1 << j:
                continue
            prev = S ^ (1 << j)
            for k in range(n):
                if prev & (1 << k):
                    cand = C[prev, k] + D[k + 1, j + 1]
                    if cand < C[S, j]:
                        C[S, j], parent[S, j] = cand, k
    j = int(np.argmin(C[full]))
    length = float(C[full, j])
    order, S = [], full
    while j != -1:
        order.append(j + 1)
        S, j = S ^ (1 << j), parent[S, j]
    return order[::-1], length


def nearest_neighbor(D):
    """Heuristic: from the current node, always walk to the closest unvisited place."""
    unvisited, cur, order = set(range(1, len(D))), 0, []
    while unvisited:
        cur = min(unvisited, key=lambda j: D[cur, j])
        order.append(cur)
        unvisited.remove(cur)
    return order, path_length(D, order)


def two_opt(D, order):
    """Improve an order by reversing segments while that shortens the path.

    Reversing order[a..b] replaces edges (prev, order[a]) and (order[b], next)
    with (prev, order[b]) and (order[a], next); the start node 0 stays fixed
    and the last edge is missing when b is the final position (open path).
    """
    order = list(order)
    nodes = [0, *order]
    improved = True
    while improved:
        improved = False
        for a in range(1, len(nodes) - 1):
            for b in range(a + 1, len(nodes)):
                before = D[nodes[a - 1], nodes[a]] + (D[nodes[b], nodes[b + 1]] if b + 1 < len(nodes) else 0)
                after = D[nodes[a - 1], nodes[b]] + (D[nodes[a], nodes[b + 1]] if b + 1 < len(nodes) else 0)
                if after < before - 1e-12:
                    nodes[a:b + 1] = nodes[a:b + 1][::-1]
                    improved = True
    return nodes[1:], path_length(D, nodes[1:])


def nn_two_opt(D):
    """Nearest neighbor followed by 2-opt."""
    order, _ = nearest_neighbor(D)
    return two_opt(D, order)


def schedule(D, order, start_min, minutes_per_unit, stay, earliest):
    """Arrival, visit start and open/closed for each stop of `order` (nodes 1..N).

    Walking a leg of length D[a, b] takes D[a, b] * minutes_per_unit minutes.
    earliest(j, t) is the first minute >= t at which place node j can be visited
    for the whole stay (waiting for it to open), or None if it cannot be: the
    visit then counts as closed and starts on arrival.

    Returns (arrive, begin, closed): three lists in visiting order.
    """
    arrive, begin, closed = [], [], []
    t, prev = float(start_min), 0
    for j in order:
        a = t + D[prev, j] * minutes_per_unit
        s = earliest(j, a)
        arrive.append(a)
        begin.append(a if s is None else s)
        closed.append(s is None)
        t, prev = begin[-1] + stay, j
    return arrive, begin, closed


def held_karp_hours(D, start_min, minutes_per_unit, stay, earliest):
    """Exact visiting order with opening hours: fewest closed visits, then earliest finish.

    Held-Karp over the same states (S, j), but a state keeps every label
    (closed, end) that no other label beats on both: `closed` visits so far and
    the minute the visit to j ends. One number per state is not enough, because
    a path that is later now may still avoid a closure that an earlier path
    cannot (and vice versa).

        arrive = end(k) + D[k][j] * minutes_per_unit
        begin  = earliest(j, arrive), or arrive with closed + 1 when None
        end(j) = begin + stay

    Arriving earlier never hurts (one can always wait), so dropping dominated
    labels keeps the optimum. With every place open the finish time is the start
    time + walking + stays, so the order is the same as held_karp(D).

    Returns (order, closed): order lists nodes 1..N, closed is the number of
    closed visits in it.
    """
    n = len(D) - 1
    if n == 0:
        return [], 0
    full = (1 << n) - 1
    # labels[S][j]: list of (closed, end, k, index of the parent label in labels[S - {j}][k])
    labels = [[None] * n for _ in range(1 << n)]

    def step(t, k, j):
        a = t + D[k, j] * minutes_per_unit
        s = earliest(j, a)
        return (1, a + stay) if s is None else (0, s + stay)

    for j in range(n):
        c, end = step(float(start_min), 0, j + 1)
        labels[1 << j][j] = [(c, end, -1, -1)]
    for S in range(1, full + 1):
        for j in range(n):
            if not S & (1 << j) or S == 1 << j:
                continue
            prev = S ^ (1 << j)
            cand = []
            for k in range(n):
                if not prev & (1 << k):
                    continue
                for idx, (c0, t0, _, _) in enumerate(labels[prev][k]):
                    c, end = step(t0, k + 1, j + 1)
                    cand.append((c0 + c, end, k, idx))
            # keep the Pareto front: sorted by (closed, end), a label survives only
            # if it ends earlier than every label with fewer or equal closures
            cand.sort()
            front, best_end = [], np.inf
            for lab in cand:
                if lab[1] < best_end - 1e-9:
                    front.append(lab)
                    best_end = lab[1]
            labels[S][j] = front
    # best final label: fewest closed, then earliest end
    j, idx = min(((j, i) for j in range(n) for i in range(len(labels[full][j]))),
                 key=lambda ji: labels[full][ji[0]][ji[1]][:2])
    closed = labels[full][j][idx][0]
    order, S = [], full
    while j != -1:
        order.append(j + 1)
        _, _, k, pidx = labels[S][j][idx]
        S, j, idx = S ^ (1 << j), k, pidx
    return order[::-1], closed
