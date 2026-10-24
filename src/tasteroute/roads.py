"""Walking distances and paths along real streets (OpenStreetMap).

Straight-line distance (geo.haversine) cuts through buildings and blocks. Here
the walkable street network is a weighted graph (intersections and bends are
nodes, street segments are edges weighted by length) and the distance between
two places is the shortest path on it, found with Dijkstra's algorithm.

The network is downloaded once per area with OSMnx and cached under data/osm/
(OpenStreetMap data, ODbL; not committed). Building a RoadNetwork from arrays
needs no download, which is how the tests use it.
"""
import hashlib
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
    def download(cls, north, south, east, west, cache_dir=CACHE_DIR):
        """Walkable streets inside a lat/lon box, from cache or from OpenStreetMap.

        osmnx.graph_from_bbox(bbox=(left, bottom, right, top), network_type="walk",
        simplify=False) queries the Overpass API for every footway, sidewalk
        and street a pedestrian may use. simplify=False keeps every bend as a
        node so a path drawn through the nodes follows the street shape.
        """
        cache_dir = Path(cache_dir)
        box = tuple(round(x, 4) for x in (north, south, east, west))
        path = cache_dir / f"walk_{hashlib.md5(str(box).encode()).hexdigest()[:12]}.npz"
        if path.exists():
            z = np.load(path)
            return cls(z["lat"], z["lon"], z["u"], z["v"], z["length_km"])

        import osmnx as ox  # optional dependency: pip install -e ".[demo]"

        ox.settings.cache_folder = str(cache_dir / "http")
        # the public Overpass server rate-limits busy clients; wait and retry
        for attempt in range(DOWNLOAD_TRIES):
            try:
                G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="walk", simplify=False)
                break
            except Exception as e:  # osmnx raises several HTTP/response error types
                if attempt == DOWNLOAD_TRIES - 1:
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
        np.savez_compressed(path, lat=lat, lon=lon, u=u, v=v, length_km=w)
        return cls(lat, lon, u, v, w)

    @classmethod
    def around(cls, lats, lons, margin_km=0.5, cache_dir=CACHE_DIR):
        """The network covering all given points plus a margin for detours."""
        lats, lons = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
        dlat = margin_km / KM_PER_DEG_LAT
        dlon = margin_km / (KM_PER_DEG_LAT * np.cos(np.radians(lats.mean())))
        return cls.download(lats.max() + dlat, lats.min() - dlat, lons.max() + dlon, lons.min() - dlon, cache_dir)

    def nearest_node(self, lat, lon):
        """Index of the street node closest (straight line) to each point."""
        lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
        return np.array([np.argmin(haversine(a, b, self.lat, self.lon)) for a, b in zip(lat, lon)])

    def routes(self, lat, lon):
        """Shortest walking routes between every pair of the given points."""
        return Routes(self, lat, lon)


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
