from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EARTH_RADIUS_M = 6_371_000.0

# Fusion strategies, ordered best-general-purpose first. Exposed so the CLI and the
# benchmark console can offer exactly the set the library implements.
FUSION_METHODS: tuple[str, ...] = ("cluster", "softmax", "median", "weighted_mean", "top1")
DEFAULT_FUSION_METHOD = "cluster"
DEFAULT_SOFTMAX_TEMP = 0.02
DEFAULT_CLUSTER_RADIUS_M = 150.0


@dataclass(frozen=True)
class GeoPose:
    latitude: float
    longitude: float
    altitude_m: float | None = None
    heading_deg: float | None = None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


# --------------------------------------------------------------------------- #
# local planar projection (equirectangular about a reference point)
# --------------------------------------------------------------------------- #
def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """WGS-84 series for metres per degree of latitude and longitude at ``lat_deg``."""
    lat = math.radians(lat_deg)
    m_lat = 111_132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_lon = 111_412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return m_lat, m_lon


def _to_local_xy(
    latitudes: np.ndarray, longitudes: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    m_lat, m_lon = _meters_per_degree(lat0)
    x = (np.asarray(longitudes, dtype=np.float64) - lon0) * m_lon
    y = (np.asarray(latitudes, dtype=np.float64) - lat0) * m_lat
    return x, y


def _from_local_xy(x: float, y: float, lat0: float, lon0: float) -> tuple[float, float]:
    m_lat, m_lon = _meters_per_degree(lat0)
    return lat0 + y / m_lat, lon0 + x / m_lon


def _sphere_mean(
    latitudes: np.ndarray, longitudes: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Weighted mean of lat/lon on the unit sphere, wrap-safe."""
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    x_mean = float(np.dot(weights, x))
    y_mean = float(np.dot(weights, y))
    z_mean = float(np.dot(weights, z))
    norm = math.sqrt(x_mean**2 + y_mean**2 + z_mean**2) or 1.0
    lat = math.degrees(math.asin(np.clip(z_mean / norm, -1.0, 1.0)))
    lon = math.degrees(math.atan2(y_mean, x_mean))
    return lat, lon


def _normalized(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=np.float64), 1e-8, None)
    total = weights.sum()
    return weights / total if total else np.full_like(weights, 1.0 / len(weights))


def _softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    """exp(score / T), shifted for numerical stability, normalized to sum 1.

    Retrieval cosine scores sit in a narrow band (~0.6-0.8) with ~0.02 gaps, so a
    linear normalization barely distinguishes rank 1 from rank 5. A small
    temperature turns those gaps into a decisive weighting.
    """
    scores = np.asarray(scores, dtype=np.float64)
    shifted = (scores - scores.max()) / max(temperature, 1e-6)
    exp = np.exp(shifted)
    total = exp.sum()
    return exp / total if total else np.full_like(exp, 1.0 / len(exp))


def _geometric_median_xy(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, iterations: int = 64
) -> tuple[float, float]:
    """Weiszfeld's algorithm: the point minimizing the weighted sum of distances.

    Robust to a minority of outliers without needing a distance threshold.
    """
    px = float(np.dot(weights, x))
    py = float(np.dot(weights, y))
    for _ in range(iterations):
        dist = np.hypot(x - px, y - py)
        near = dist < 1e-6
        if near.any():
            return float(x[near][0]), float(y[near][0])
        inv = weights / dist
        denom = inv.sum()
        if not denom:
            break
        nx = float(np.dot(inv, x) / denom)
        ny = float(np.dot(inv, y) / denom)
        if math.hypot(nx - px, ny - py) < 1e-4:
            px, py = nx, ny
            break
        px, py = nx, ny
    return px, py


def _dominant_cluster(
    x: np.ndarray, y: np.ndarray, softmax_w: np.ndarray, radius_m: float
) -> np.ndarray:
    """Greedy single-linkage clustering of the candidates in the local plane.

    Returns the indices of the cluster carrying the most softmax weight - i.e. the
    location the strongest matches agree on, with stray candidates dropped.
    """
    n = len(x)
    order = np.argsort(-softmax_w)
    assigned = np.full(n, -1, dtype=int)
    clusters: list[list[int]] = []
    for seed in order:
        if assigned[seed] != -1:
            continue
        cid = len(clusters)
        members = [int(seed)]
        assigned[seed] = cid
        # absorb every still-free candidate within radius of any member (single-linkage)
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if assigned[j] != -1:
                    continue
                if any(math.hypot(x[j] - x[m], y[j] - y[m]) <= radius_m for m in members):
                    assigned[j] = cid
                    members.append(j)
                    changed = True
        clusters.append(members)
    best = max(clusters, key=lambda members: float(softmax_w[members].sum()))
    return np.asarray(best, dtype=int)


def fuse_geo(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    weights: np.ndarray,
    *,
    method: str = DEFAULT_FUSION_METHOD,
    altitudes: np.ndarray | None = None,
    softmax_temp: float = DEFAULT_SOFTMAX_TEMP,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
) -> GeoPose:
    """Fuse ranked geographic matches into one pose.

    ``weights`` are the retrieval similarity scores of the matches, best first.

    method
        ``"weighted_mean"`` - legacy: unit-sphere mean, weight = normalized score.
        ``"softmax"``       - unit-sphere mean, weight = softmax(score / temp).
        ``"median"``        - weighted geometric median (Weiszfeld), outlier-robust.
        ``"cluster"``       - keep the largest geo-consistent cluster of the
                              candidates, then softmax-fuse only that cluster.
                              This is the default: it discards the stray matches
                              that otherwise drag a plain average hundreds of
                              metres off.
        ``"top1"``          - the single best match, no fusion.
    """
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if latitudes.size == 0:
        return GeoPose(latitude=float("nan"), longitude=float("nan"))

    if method not in FUSION_METHODS:
        raise ValueError(f"Unknown fusion method {method!r}; expected one of {FUSION_METHODS}")

    def _altitude(mask: np.ndarray, w: np.ndarray) -> float | None:
        if altitudes is None:
            return None
        alt = np.asarray(altitudes, dtype=np.float64)[mask]
        finite = np.isfinite(alt)
        if not finite.any():
            return None
        w = w[finite]
        w = w / w.sum() if w.sum() else np.full(finite.sum(), 1.0 / finite.sum())
        return float(np.dot(w, alt[finite]))

    full_mask = np.ones(latitudes.size, dtype=bool)

    if method == "top1":
        best = int(np.argmax(weights)) if weights.size else 0
        one = np.zeros(latitudes.size, dtype=bool)
        one[best] = True
        return GeoPose(
            latitude=float(latitudes[best]),
            longitude=float(longitudes[best]),
            altitude_m=_altitude(one, np.array([1.0])),
        )

    if method == "weighted_mean":
        w = _normalized(weights)
        lat, lon = _sphere_mean(latitudes, longitudes, w)
        return GeoPose(latitude=lat, longitude=lon, altitude_m=_altitude(full_mask, w))

    if method == "softmax":
        w = _softmax_weights(weights, softmax_temp)
        lat, lon = _sphere_mean(latitudes, longitudes, w)
        return GeoPose(latitude=lat, longitude=lon, altitude_m=_altitude(full_mask, w))

    lat0 = float(np.mean(latitudes))
    lon0 = float(np.mean(longitudes))
    x, y = _to_local_xy(latitudes, longitudes, lat0, lon0)

    if method == "median":
        w = _softmax_weights(weights, softmax_temp)
        mx, my = _geometric_median_xy(x, y, w)
        lat, lon = _from_local_xy(mx, my, lat0, lon0)
        return GeoPose(latitude=lat, longitude=lon, altitude_m=_altitude(full_mask, w))

    # method == "cluster"
    softmax_all = _softmax_weights(weights, softmax_temp)
    keep = _dominant_cluster(x, y, softmax_all, cluster_radius_m)
    sub_w = _softmax_weights(weights[keep], softmax_temp)
    lat, lon = _sphere_mean(latitudes[keep], longitudes[keep], sub_w)
    mask = np.zeros(latitudes.size, dtype=bool)
    mask[keep] = True
    return GeoPose(latitude=lat, longitude=lon, altitude_m=_altitude(mask, sub_w))


def weighted_geo_fusion(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    weights: np.ndarray,
    altitudes: np.ndarray | None = None,
    method: str = DEFAULT_FUSION_METHOD,
    softmax_temp: float = DEFAULT_SOFTMAX_TEMP,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
) -> GeoPose:
    """Backwards-compatible entry point. Delegates to :func:`fuse_geo`.

    The default strategy is ``"cluster"`` rather than the historical plain
    score-weighted mean, because on wide-area maps a single stray top-5 match
    pulls the mean far enough to blow past the 100 m error band.
    """
    return fuse_geo(
        latitudes,
        longitudes,
        weights,
        method=method,
        altitudes=altitudes,
        softmax_temp=softmax_temp,
        cluster_radius_m=cluster_radius_m,
    )


# --------------------------------------------------------------------------- #
# search window around a position prior
# --------------------------------------------------------------------------- #
def search_window(
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    min_keep: int = 1,
) -> np.ndarray:
    """Boolean mask of the reference tiles retrieval may return, given a prior.

    In flight the navigation filter always has a predicted position and an
    uncertainty, so retrieval only has to discriminate between tiles that are
    plausible under that prior; distant look-alikes are never candidates.
    ``radius_m`` is usually ``k * sigma``. The ``min_keep`` tiles nearest the
    centre are always kept, so a prior tighter than the tile spacing still leaves
    something to choose from.
    """
    x, y = _to_local_xy(ref_lat, ref_lon, center_lat, center_lon)
    dist = np.hypot(x, y)
    mask = dist <= radius_m
    if int(mask.sum()) < min_keep:
        mask[np.argsort(dist)[:min_keep]] = True
    return mask
