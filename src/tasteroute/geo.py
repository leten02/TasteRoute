"""Distances on the sphere and each user's geographic anchor.

Straight-line (great-circle) distance, used for re-ranking and the routing
experiments. Walking distance along streets is in roads.py. Proposal 6.1-6.2.
"""
import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; inputs in degrees, numpy-broadcastable.

    d = 2R arcsin(sqrt(sin^2(dphi/2) + cos(phi1) cos(phi2) sin^2(dlambda/2)))
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def distance_matrix(lats, lons):
    """N x N matrix of pairwise haversine distances (km)."""
    lats, lons = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
    return haversine(lats[:, None], lons[:, None], lats[None, :], lons[None, :])


def user_anchors(train, items, n_users):
    """Each user's anchor: the median lat/lon of the places they rated in train.

    The median, unlike the mean, is not pulled away by one far-off trip.
    Users with no training ratings get NaN.

    Returns:
        (n_users, 2) array of (lat, lon).
    """
    lat = items.set_index("i").lat.reindex(range(items.i.max() + 1)).values
    lon = items.set_index("i").lon.reindex(range(items.i.max() + 1)).values
    df = train[["u", "i"]].assign(lat=lat[train.i.values], lon=lon[train.i.values])
    med = df.groupby("u")[["lat", "lon"]].median()
    out = np.full((n_users, 2), np.nan)
    out[med.index.values] = med.values
    return out
