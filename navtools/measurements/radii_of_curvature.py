'''
|===================================== radii_of_curvature.py ======================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/measurements/gravity/radii_of_curvature.py                                   |
|  @brief    Calculations for common Earth radii used for navigation.                              |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from navtools.constants import WGS84_R0, WGS84_E2

# === TRANSVERSERADIUS ===
@njit(cache=True, fastmath=True)
def transverseRadius(phi: np.float64) -> np.float64:
  """Calculates the transverse radius relative to user latitude

  Parameters
  ----------
  phi : np.float64
      Latitude [rad]

  Returns
  -------
  np.float64
      Earth's transverse radius at Latitude
  """
  sinPhi2 = np.sin(phi)**2
  t = 1 - WGS84_E2*sinPhi2
  return WGS84_R0 / np.sqrt(t)


# === MERIDIANRADIUS ===
@njit(cache=True, fastmath=True)
def meridianRadius(phi: np.float64) -> np.float64:
  """Calculates the meridian radius relative to user latitude

  Parameters
  ----------
  phi : np.float64
      Latitude [rad]

  Returns
  -------
  np.float64
      Earth's meridian radius at Latitude
  """
  sinPhi2 = np.sin(phi)**2
  t = 1 - WGS84_E2*sinPhi2
  return WGS84_R0 * (1 - WGS84_E2) / (t**1.5)


# === GEOCENTRICRADIUS ===
@njit(cache=True, fastmath=True)
def geocentricRadius(phi: np.float64) -> np.float64:
  """Calculates the geocentric radius relative to the user latitude

  Parameters
  ----------
  phi : np.float64
      Latitude [rad]

  Returns
  -------
  np.float64
      Earth's geocentric radius at Latitude
  """
  sinPhi2 = np.sin(phi)**2
  cosPhi2 = np.cos(phi)**2
  t = 1 - WGS84_E2*sinPhi2
  Re = WGS84_R0 / np.sqrt(t)
  return Re * np.sqrt(cosPhi2 + (1 - WGS84_E2)**2 * sinPhi2)


# === RADIIOFCURVATURE ===
@njit(cache=True, fastmath=True)
def radiiOfCurvature(phi: np.float64) -> tuple[np.float64, np.float64, np.float64]:
  """Calculates the transverse, meridian, and geocentric radii or curvature

  Parameters
  ----------
  phi : _type_
      Latitude [rad]

  Returns
  -------
  tuple[np.float64, np.float64, np.float64]
      Re:     Earth's transverse radius at Latitude [m]
      Rn:     Earth's meridian radius at Latitude [m]
      r_es_e: Earth's geocentric radius at Latitude [m]
  """
  sinPhi2 = np.sin(phi)**2
  cosPhi2 = np.cos(phi)**2
  t = 1 - WGS84_E2*sinPhi2
  Re = WGS84_R0 / np.sqrt(t)
  Rn = WGS84_R0 * (1- WGS84_E2) / (t**1.5)
  r_es_e = Re * np.sqrt(cosPhi2 + (1 - WGS84_E2)**2 * sinPhi2)
  return Re, Rn, r_es_e
  