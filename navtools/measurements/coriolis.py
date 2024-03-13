"""
|========================================== coriolis.py ===========================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/measurements/gravity/coriolis.py                                             |
|  @brief    Calculations for common Earth rotation rates used for navigation frame.               |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
"""

__all__ = ["earthRate", "transportRate", "coriolisRate"]

import numpy as np
from numba import njit
from navtools.conversions.skewmat import skew
from navtools.constants import GNSS_OMEGA_EARTH
from .radii_of_curvature import radiiOfCurvature


# === EARTHRATE ===
@njit(cache=True, fastmath=True)
def earthRate(lla: np.ndarray) -> np.ndarray:
    """Rotation rate of the earth relative to the 'NAV' frame

    Parameters
    ----------
    lla : np.ndarray
        3x1 Latitude, longitude, height [rad, rad, m]

    Returns
    -------
    np.ndarray
        3x3 Skew symmetric form of the earth'r rotation in the 'NAV' frame
    """
    sinPhi = np.sin(lla[0])
    cosPhi = np.cos(lla[0])
    return skew(
        np.array(
            [GNSS_OMEGA_EARTH * cosPhi, 0.0, GNSS_OMEGA_EARTH * sinPhi], dtype=np.double
        )
    )


# === TRANSPORTRATE ===
@njit(cache=True, fastmath=True)
def transportRate(lla: np.ndarray, v_nb_n: np.ndarray) -> np.ndarray:
    """Transport rate of the 'ECEF' frame relative to the 'NAV' frame

    Parameters
    ----------
    lla : np.ndarray
        3x1 Latitude, longitude, height [rad, rad, m]
    v_nb_n : np.ndarray
        3x1 Velocity in the 'NED' coordinate system

    Returns
    -------
    np.ndarray
        3x3 Skew symmetric form of the earth'r rotation in the 'NAV' frame
    """
    phi, lam, h = lla
    vn, ve, vd = v_nb_n
    Re, Rn, r_es_e = radiiOfCurvature(phi)
    return skew(
        np.array(
            [ve / (Re + h), -vn / (Rn + h), -ve * np.tan(phi) / (Re + h)],
            dtype=np.double,
        )
    )


# === CORIOLIS ===
@njit(cache=True, fastmath=True)
def coriolisRate(lla: np.ndarray, v_nb_n: np.ndarray) -> np.ndarray:
    """Coriolis effect perceived in the nave frame

    Parameters
    ----------
    lla : np.ndarray
        3x1 Latitude, longitude, height [rad, rad, m]
    v_nb_n : np.ndarray
        3x1 Velocity in the 'NED' coordinate system

    Returns
    -------
    np.ndarray
        3x1 Coriolis effect
    """
    W_ie_n = earthRate(lla)
    W_en_n = transportRate(lla, v_nb_n)
    return (W_en_n + 2 * W_ie_n) @ v_nb_n
