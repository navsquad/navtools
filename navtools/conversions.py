import numpy as np
from numba import njit
from collections import namedtuple

from navtools.constants import WGS84_RADIUS, WGS84_ECCENTRICITY

### reference frames ###
ECEF = namedtuple("ECEF", ["x", "y", "z"])
ENU = namedtuple("ENU", ["east", "north", "up"])
GEODETIC = namedtuple("GEODETIC", ["lat", "lon", "alt"])


@njit
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


@njit
def lla2ecef(lat: np.array, lon: np.array, alt: np.array, degrees=True) -> ECEF:
    if degrees:
        lat = np.radians(lat)
        lon = np.radians(lon)

    den = 1 - WGS84_ECCENTRICITY**2 * np.sin(lat) ** 2
    RN = WGS84_RADIUS / np.sqrt(den)

    x = (RN + alt) * np.cos(lat) * np.cos(lon)
    y = (RN + alt) * np.cos(lat) * np.sin(lon)
    z = (RN * (1 - WGS84_ECCENTRICITY**2) + alt) * np.sin(lat)

    return ECEF(x=x, y=y, z=z)


@njit
def ecef2enu(
    x: np.array,
    y: np.array,
    z: np.array,
    lat0: float,
    lon0: float,
    alt0: float,
    degrees=True,
) -> ENU:
    if degrees:
        lat0 = np.radians(lat0)
        lon0 = np.radians(lon0)

    R = np.array(
        [
            [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
            [-np.sin(lon0), np.cos(lon0), 0],
            [-np.cos(lat0) * np.cos(lon0), -np.cos(lat0) * np.sin(lon0), -np.sin(lat0)],
        ],
    )

    ecef0 = lla2ecef(lat=lat0, lon=lon0, alt=alt0, degrees=False)
    rel_x_pos = x - ecef0.x
    rel_y_pos = y - ecef0.y
    rel_z_pos = z - ecef0.z
    rel_pos = np.array([rel_x_pos, rel_y_pos, rel_z_pos], dtype=np.float64).T

    ned = R @ rel_pos

    return ENU(east=ned[1], north=ned[0], up=-ned[2])


@njit
def enu2ecefv(
    east: np.array, north: np.array, up: np.array, lat0, lon0, degrees=True
) -> ECEF:
    enu = np.array([east, north, up], dtype=np.float64).T

    if degrees:
        lat0 = np.radians(lat0)
        lon0 = np.radians(lon0)

    R = np.array(
        [
            [-np.sin(lon0), -np.cos(lon0) * np.sin(lat0), np.cos(lon0) * np.cos(lat0)],
            [np.cos(lon0), -np.sin(lon0) * np.sin(lat0), np.sin(lon0) * np.cos(lat0)],
            [0, np.cos(lat0), np.sin(lat0)],
        ],
    )

    ecef = R @ enu

    return ECEF(x=ecef[0], y=ecef[1], z=ecef[2])


### signals ###
@njit
def cn02snr(cn0: float, front_end_bw: float = 4e6, noise_figure: float = 0.0):
    snr = cn0 - 10 * np.log10(front_end_bw) - noise_figure  # dB

    return snr


@njit
def snr2cn0(snr: float, front_end_bw: float = 4e6, noise_figure: float = 0.0):
    cn0 = snr + 10 * np.log10(front_end_bw) + noise_figure  # dB-Hz

    return cn0
