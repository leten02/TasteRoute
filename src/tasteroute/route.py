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
