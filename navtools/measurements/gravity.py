'''
|================================== navtools/gravity/gravity.py ===================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/measurements/gravity/gravity.py                                              |
|  @brief    Calculations for Earth gravity rates in different frames.                             |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from navtools.constants import WGS84_E2, WGS84_R0, WGS84_RP, WGS84_F, WGS84_MU, GNSS_OMEGA_EARTH, WGS84_J2
from navtools.conversions.coordinates import ned2ecefDcm
from navtools.conversions.coordinates import ecef2lla


# === SOMIGLIANA ===
@njit(cache=True, fastmath=True)
def somigliana(phi: np.float64) -> np.float64:
  """Calculates the somilgiana model to calculate reference gravity

  Parameters
  ----------
  phi : np.float64
      Latitude, longitude, height [rad, rad, m]

  Returns
  -------
  np.float64
      somigliana model gravity
  """
  sinPhi2 = np.sin(phi)**2
  return 9.7803253359 * ((1 + 0.001931853*sinPhi2) / np.sqrt(1 - WGS84_E2*sinPhi2))


# === NEDG ===
@njit(cache=True, fastmath=True)
def nedg(lla: np.ndarray) -> np.ndarray:
  """Calculates gravity in the 'NED' frame

  Parameters
  ----------
  lla : np.ndarray
      Latitude, longitude, height [rad, rad, m]

  Returns
  -------
  np.ndarray
      'NED' gravity
  """
  phi, lam, h = lla
  sinPhi2 = np.sin(phi)**2
  g0 = 9.7803253359 * ((1 + 0.001931853*sinPhi2) / np.sqrt(1 - WGS84_E2*sinPhi2))
  return np.array([-8.08e-9 * h * np.sin(2*phi), \
                    0.0, \
                    g0 * (1 - (2 / WGS84_R0) * (1 + WGS84_F * (1 - 2 * sinPhi2) + \
                    (GNSS_OMEGA_EARTH**2 * WGS84_R0**2 * WGS84_RP / WGS84_MU)) * h + \
                    (3 * h**2 / WGS84_R0**2))
                  ])
  
  
# === GRAVITYECEF ===
@njit(cache=True, fastmath=True)
def ecefg(r_eb_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Calculates gravity and gravitational acceleration in the 'ECEF' frame

  Parameters
  ----------
  r_eb_e : np.ndarray
      Latitude, longitude, height [rad, rad, m]

  Returns
  -------
  tuple[np.ndarray, np.ndarray]
      g:     'ECEF' gravity
      gamma: 'ECEF' gravitational acceleration
  """
  x,y,z = r_eb_e
  mag_r = np.linalg.norm(r_eb_e)
  if mag_r == 0.0:
    gamma = np.zeros(3)
    g = np.zeros(3)
  else:
    zeta = 5 * (z / mag_r)**2
    M = np.array([(1.0 - zeta) * x, \
                  (1.0 - zeta) * y, \
                  (3.0 - zeta) * z], 
                 dtype=np.double)
    gamma = -WGS84_MU / mag_r**3 * (r_eb_e + 1.5 * WGS84_J2 * (WGS84_R0 / mag_r)**2 * M)
    g = gamma + GNSS_OMEGA_EARTH*GNSS_OMEGA_EARTH * np.array([x, y, 0.0], dtype=np.double)
  return g, gamma


# === NED2ECEFG ===
@njit(cache=True, fastmath=True)
def ned2ecefg(r_eb_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Calculates gravity in the 'NAV' frame and rotates it to the 'ECEF' frame

  Parameters
  ----------
  r_eb_e : np.ndarray
      Latitude, longitude, height [rad, rad, m]

  Returns
  -------
  tuple[np.ndarray, np.ndarray]
      g:     'ECEF' gravity
      gamma: 'ECEF' gravitational acceleration
  """
  lla = ecef2lla(r_eb_e)
  g_ned = nedg(lla)
  C_n_e = ned2ecefDcm(lla)
  g_ecef = C_n_e @ g_ned
  gamma = g_ecef - GNSS_OMEGA_EARTH*GNSS_OMEGA_EARTH * np.array([r_eb_e[0], r_eb_e[1], 0.0], dtype=np.double)
  return g_ecef, gamma
