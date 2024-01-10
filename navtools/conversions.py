import numpy as np
from numba import njit, vectorize, float64
from collections import namedtuple

from numpy import sin, cos
from navtools.constants import WGS84_RADIUS, WGS84_ECCENTRICITY

# Reference Frames
ECEF = namedtuple("ECEF", ["x", "y", "z"])
ENU = namedtuple("ENU", ["east", "north", "up"])
GEODETIC = namedtuple("GEODETIC", ["lat", "lon", "alt"])


@njit(cache=True)
def ecef2lla(x: np.array, y: np.array, z: np.array) -> GEODETIC:
    # TODO: clean this up a little
    p = (x**2 + y**2) / WGS84_RADIUS**2
    q = ((1 - WGS84_ECCENTRICITY**2) * z**2) / WGS84_RADIUS**2
    r = (p + q - WGS84_ECCENTRICITY**4) / 6
    s = (WGS84_ECCENTRICITY**4 * p * q) / (4 * r**3)
    t = (1 + s + np.sqrt(s * (2 + s))) ** (1 / 3)
    u = r * (1 + t + (1 / t))
    v = np.sqrt(u**2 + q * WGS84_ECCENTRICITY**4)
    w = WGS84_ECCENTRICITY**2 * (u + v - q) / (2 * v)
    k = np.sqrt(u + v + w**2) - w

    D = k * np.sqrt(x**2 + y**2) / (k + WGS84_ECCENTRICITY**2)

    lat = np.degrees(np.arctan2(z, D))
    lon = np.degrees(np.arctan2(y, x))
    alt = (k + WGS84_ECCENTRICITY**2 - 1) / k * (np.sqrt(D**2 + z**2))

    return GEODETIC(lat=lat, lon=lon, alt=alt)


@njit(cache=True)
def lla2ecef(lat: np.array, lon: np.array, alt: np.array, deg=True) -> ECEF:
    if deg:
        lat = np.radians(lat)
        lon = np.radians(lon)

    den = 1 - WGS84_ECCENTRICITY**2 * sin(lat) ** 2
    RN = WGS84_RADIUS / np.sqrt(den)

    x = (RN + alt) * cos(lat) * cos(lon)
    y = (RN + alt) * cos(lat) * sin(lon)
    z = (RN * (1 - WGS84_ECCENTRICITY**2) + alt) * sin(lat)

    return ECEF(x=x, y=y, z=z)


@njit(cache=True)
def uvw2enu(u: float, v: float, w: float, lat0: float, lon0: float, deg=True):
    if deg:
        lat0 = np.radians(lat0)
        lon0 = np.radians(lon0)

    t = cos(lon0) * u + sin(lon0) * v
    east = -sin(lon0) * u + cos(lon0) * v
    north = -sin(lat0) * t + cos(lat0) * w
    up = cos(lat0) * t + sin(lat0) * w

    return ENU(east=east, north=north, up=up)


@njit(cache=True)
def enu2uvw(
    east: float, north: float, up: float, lat0: float, lon0: float, deg: bool = True
):
    t = cos(lat0) * up - sin(lat0) * north
    w = sin(lat0) * up + cos(lat0) * north

    u = cos(lon0) * t - sin(lon0) * east
    v = sin(lon0) * t + cos(lon0) * east

    return ECEF(x=u, y=v, z=w)


@njit(cache=True)
def ecef2enu(
    x: float,
    y: float,
    z: float,
    lat0: float,
    lon0: float,
    alt0: float,
    deg: bool = True,
):
    x0, y0, z0 = lla2ecef(lat=lat0, lon=lon0, alt=alt0, deg=deg)

    enu = uvw2enu(x - x0, y - y0, z - z0, lat0, lon0, deg=deg)

    return ENU(east=enu.east, north=enu.north, up=enu.up)


@njit(cache=True)
def enu2ecef(
    east: float,
    north: float,
    up: float,
    lat0: float,
    lon0: float,
    h0: float,
    deg: bool = True,
):
    x0, y0, z0 = lla2ecef(lat0, lon0, h0, deg=deg)
    dx, dy, dz = enu2uvw(east, north, up, lat0, lon0, deg=deg)

    return ECEF(x=x0 + dx, y=y0 + dy, z=z0 + dz)


# Signals
@njit(cache=True)
def cn02snr(cn0: float, front_end_bw: float = 4e6, noise_figure: float = 0.0):
    snr = cn0 - 10 * np.log10(front_end_bw) - noise_figure  # dB

    return snr


@njit(cache=True)
def snr2cn0(snr: float, front_end_bw: float = 4e6, noise_figure: float = 0.0):
    cn0 = snr + 10 * np.log10(front_end_bw) + noise_figure  # dB-Hz

    return cn0
