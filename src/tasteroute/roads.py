"""Walking distances and paths along real streets (OpenStreetMap).

Straight-line distance (geo.haversine) cuts through buildings and blocks. Here
the walkable street network is a weighted graph (intersections and bends are
nodes, street segments are edges weighted by length) and the distance between
two places is the shortest path on it, found with Dijkstra's algorithm.

The network is downloaded once per area with OSMnx and cached under data/osm/
(OpenStreetMap data, ODbL; not committed). A cached network is reused for any
request whose box it covers, so downloading Center City once up front
(python -m tasteroute.roads) serves every route there without waiting.
Building a RoadNetwork from arrays needs no download, which is how the tests
use it.
"""
import hashlib
import os
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .geo import haversine

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "osm"
KM_PER_DEG_LAT = 111.32
WALK_KMH = 4.5
DOWNLOAD_TRIES = 3
REQUEST_TIMEOUT = 180  # seconds per Overpass request
# Center City Philadelphia plus University City and Fishtown edges: (north, south, east, west)
CENTER_CITY_BOX = (39.978, 39.928, -75.128, -75.205)


class RoadNetwork:
    """A directed street graph: node coordinates and edges with lengths in km."""

    def __init__(self, lat, lon, u, v, length_km):
        self.lat = np.asarray(lat, dtype=float)
        self.lon = np.asarray(lon, dtype=float)
        u, v = np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)
        w = np.asarray(length_km, dtype=float)
        # keep only the shortest of parallel edges between the same two nodes
        n = len(self.lat)
        key = u * n + v
        order = np.lexsort((w, key))  # sort by key, then by length
        first = np.ones(len(order), dtype=bool)
        first[1:] = key[order][1:] != key[order][:-1]
        keep = order[first]
        # csr_matrix((data, (row, col)), shape) builds a sparse matrix from
        # (row, col, value) triples: graph[a, b] = length of the street a -> b.
        self.graph = csr_matrix((w[keep], (u[keep], v[keep])), shape=(n, n))

    @classmethod
    def download(cls, north, south, east, west, cache_dir=CACHE_DIR, tries=DOWNLOAD_TRIES, timeout=REQUEST_TIMEOUT):
        """Walkable streets inside a lat/lon box, from cache or from OpenStreetMap.

        osmnx.graph_from_bbox(bbox=(left, bottom, right, top), network_type="walk",
        simplify=False) queries the Overpass API for every footway, sidewalk
        and street a pedestrian may use. simplify=False keeps every bend as a
        node so a path drawn through the nodes follows the street shape.
        tries and timeout (seconds per request) bound how long a busy server
        can make the caller wait; an interactive app should keep both small.
        tries=0 means cache only: raise right away if the area is not cached.
        """
        cache_dir = Path(cache_dir)
        box = tuple(round(x, 4) for x in (north, south, east, west))
        path = cache_dir / f"walk_{hashlib.md5(str(box).encode()).hexdigest()[:12]}.npz"
        # same box first (older cache files have no `box` entry), then any covering box
        for cached in (path if path.exists() else None, cls._cached_covering(cache_dir, box)):
            if cached is None:
                continue
            try:
                with np.load(cached) as z:
                    return cls(z["lat"], z["lon"], z["u"], z["v"], z["length_km"])
            except (OSError, ValueError, KeyError):  # damaged file: ignore it
                continue
        if tries == 0:
            raise RuntimeError("this area's walking network is not cached")

        import osmnx as ox  # optional dependency: pip install -e ".[demo]"

        ox.settings.cache_folder = str(cache_dir / "http")
        ox.settings.requests_timeout = timeout
        # the public Overpass server rate-limits busy clients; wait and retry
        for attempt in range(tries):
            try:
                G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="walk", simplify=False)
                break
            except Exception as e:  # osmnx raises several HTTP/response error types
                if attempt == tries - 1:
                    raise RuntimeError("OpenStreetMap download failed; retry later or use --straight") from e
                wait = 30 * 2 ** attempt
                print(f"OpenStreetMap busy ({type(e).__name__}), retrying in {wait}s", flush=True)
                time.sleep(wait)
        ids = np.array(list(G.nodes))
        index = {node: k for k, node in enumerate(ids)}
        lat = np.array([G.nodes[node]["y"] for node in ids])
        lon = np.array([G.nodes[node]["x"] for node in ids])
        edges = np.array([(index[a], index[b], d["length"] / 1000) for a, b, d in G.edges(data=True)])
        u, v, w = edges[:, 0].astype(int), edges[:, 1].astype(int), edges[:, 2]
        cache_dir.mkdir(parents=True, exist_ok=True)
        # write to a temporary name, then os.replace (atomic rename): an interrupted
        # save never leaves a half-written walk_*.npz behind
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, lat=lat, lon=lon, u=u, v=v, length_km=w, box=np.array(box))
        os.replace(tmp, path)
        return cls(lat, lon, u, v, w)

    @staticmethod
    def _cached_covering(cache_dir, box):
        """Smallest cached network file whose box contains `box`, or None."""
        north, south, east, west = box
        best = None
        for f in sorted(Path(cache_dir).glob("walk_*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            try:
                with np.load(f) as z:  # np.load on .npz reads each array only when accessed
                    if "box" not in z.files:
                        continue
                    n, s, e, w = z["box"]
            except (OSError, ValueError):  # damaged file: skip it instead of failing every lookup
                continue
            if n >= north and s <= south and e >= east and w <= west:
                area = (n - s) * (e - w)
                if best is None or area < best[0]:
                    best = (area, f)
        return None if best is None else best[1]

    @classmethod
    def around(cls, lats, lons, margin_km=0.5, cache_dir=CACHE_DIR, **download_kw):
        """The network covering all given points plus a margin for detours.

        download_kw (tries, timeout) are passed on to download().
        """
        lats, lons = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
        dlat = margin_km / KM_PER_DEG_LAT
        dlon = margin_km / (KM_PER_DEG_LAT * np.cos(np.radians(lats.mean())))
        return cls.download(lats.max() + dlat, lats.min() - dlat, lons.max() + dlon, lons.min() - dlon, cache_dir,
                            **download_kw)

    def nearest_node(self, lat, lon):
        """Index of the street node closest to each point.

        scipy.spatial.cKDTree(points).query(q) finds the nearest of `points` to
        each query in O(log n). Coordinates are scaled to km first (longitude
        shrinks by cos(latitude)) so nearest in the tree means nearest on the map.
        """
        lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
        if not hasattr(self, "_tree"):
            from scipy.spatial import cKDTree

            self._kx = KM_PER_DEG_LAT * np.cos(np.radians(self.lat.mean()))
            self._tree = cKDTree(np.column_stack([self.lat * KM_PER_DEG_LAT, self.lon * self._kx]))
        _, idx = self._tree.query(np.column_stack([lat * KM_PER_DEG_LAT, lon * self._kx]))
        return np.asarray(idx)

    def routes(self, lat, lon):
        """Shortest walking routes between every pair of the given points."""
        return Routes(self, lat, lon)

    def contains(self, lat, lon, margin_km=0.2):
        """True for points inside the network's node extent (shrunk by a margin).

        Points near the edge of a downloaded box can have their only route
        leave the box, so they are better left out of distance experiments.
        """
        lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        dlat = margin_km / KM_PER_DEG_LAT
        dlon = margin_km / (KM_PER_DEG_LAT * np.cos(np.radians(self.lat.mean())))
        return ((lat >= self.lat.min() + dlat) & (lat <= self.lat.max() - dlat)
                & (lon >= self.lon.min() + dlon) & (lon <= self.lon.max() - dlon))

    def distances_from(self, lat, lon, lats, lons):
        """Walking km from one point to many points (one Dijkstra run).

        Same snapping as Routes: each point joins its nearest street node by a
        straight hop. Unreachable targets get NaN.
        """
        src = self.nearest_node(lat, lon)[0]
        dst = self.nearest_node(lats, lons)
        d = dijkstra(self.graph, directed=True, indices=src)[dst]
        snap_src = haversine(lat, lon, self.lat[src], self.lon[src])
        snap_dst = haversine(np.asarray(lats), np.asarray(lons), self.lat[dst], self.lon[dst])
        out = d + snap_src + snap_dst
        return np.where(np.isfinite(out), out, np.nan)


class Routes:
    """Walking distance matrix between points and the street path for any pair.

    Each point is snapped to its nearest street node; the straight hop from the
    point to that node is added at both ends. If two points are not connected
    in the downloaded network (a gap at its edge), the straight-line distance
    is used for that pair instead.
    """

    def __init__(self, net, lat, lon):
        self.net = net
        self.lat, self.lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        self.nodes = net.nearest_node(self.lat, self.lon)
        snap = haversine(self.lat, self.lon, net.lat[self.nodes], net.lon[self.nodes])
        # dijkstra(graph, indices, return_predecessors=True): shortest distances
        # from each source in `indices` to every node, and for each node the
        # previous node on that shortest path (-9999 when unreachable).
        dist, self.pred = dijkstra(net.graph, directed=True, indices=self.nodes, return_predecessors=True)
        D = dist[:, self.nodes] + snap[:, None] + snap[None, :]
        straight = haversine(self.lat[:, None], self.lon[:, None], self.lat[None, :], self.lon[None, :])
        self.connected = np.isfinite(D)
        D = np.where(self.connected, D, straight)
        np.fill_diagonal(D, 0.0)
        self.D = D

    def path(self, a, b):
        """(lat, lon) points of the walking route from point a to point b."""
        if not self.connected[a, b]:
            return [(self.lat[a], self.lon[a]), (self.lat[b], self.lon[b])]
        nodes, k = [], self.nodes[b]
        while k >= 0:
            nodes.append(k)
            k = self.pred[a, k]
        nodes.reverse()
        return ([(self.lat[a], self.lon[a])]
                + [(self.net.lat[k], self.net.lon[k]) for k in nodes]
                + [(self.lat[b], self.lon[b])])

    def tour_path(self, order):
        """Street points for visiting point 0, then the points in `order`."""
        stops = [0, *order]
        out = [(self.lat[0], self.lon[0])] if len(stops) == 1 else []
        for a, b in zip(stops[:-1], stops[1:]):
            out.extend(self.path(a, b)[(1 if out else 0):])
        return out


if __name__ == "__main__":
    # one-time download of Center City (a few minutes), so the dashboard and the
    # demo never wait for OpenStreetMap inside it
    t = time.time()
    net = RoadNetwork.download(*CENTER_CITY_BOX)
    print(f"Center City walking network: {len(net.lat):,} nodes, {net.graph.nnz:,} street segments "
          f"({time.time() - t:.0f}s), cached in {CACHE_DIR}")
