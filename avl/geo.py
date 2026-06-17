from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EARTH_RADIUS_M = 6_371_000.0


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


def weighted_geo_fusion(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    weights: np.ndarray,
    altitudes: np.ndarray | None = None,
) -> GeoPose:
    """
    Fuse top-K geographic matches using cosine-similarity weights.

    Latitude/longitude are averaged on the unit sphere to avoid wrap artifacts.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-8, None)
    weights /= weights.sum()

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

    altitude = None
    if altitudes is not None and np.isfinite(altitudes).any():
        altitude = float(np.dot(weights, np.nan_to_num(altitudes, nan=0.0)))

    return GeoPose(latitude=lat, longitude=lon, altitude_m=altitude)
