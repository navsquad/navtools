'''
|========================================= position.py ============================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/coordinate/position.py                                                       |
|  @brief    Common position coordinate frame transformations.                                     |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from .dcm import *
from navtools.constants import WGS84_R0, WGS84_E2

#--------------------------------------------------------------------------------------------------#
# === LLA2ECI ===
@njit(cache=True, fastmath=True)
def lla2eci(lla: np.ndarray, dt: np.float64) -> np.ndarray:
  """Latitude-Longitude-Height to Earth-Centered-Inertial coordinates

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : np.float64
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECI x,y,z coordinates [m]
  """
  xyz = lla2ecef(lla)
  return ecef2eciDcm(dt) @ xyz


# === LLA2ECEF ===
@njit(cache=True, fastmath=True)
def lla2ecef(lla: np.ndarray) -> np.ndarray:
  """Latitude-Longitude-Height to Earth-Centered-Earth-Fixed coordinates

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z coordinates [m]
  """
  sinPhi, sinLam = np.sin(lla[:2])
  cosPhi, cosLam = np.cos(lla[:2])
  h = lla[2]
  
  Re = WGS84_R0 / np.sqrt(1 - WGS84_E2*sinPhi*sinPhi)
  x = (Re + h) * cosPhi*cosLam
  y = (Re + h) * cosPhi*sinLam
  z = (Re * (1 - WGS84_E2) + h) * sinPhi
  
  return np.array([x,y,z], dtype=np.double)


# === LLA2NED ===
@njit(cache=True, fastmath=True)
def lla2ned(lla: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Latitude-Longitude-Height to North-East-Down coordinates

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  lla0 : np.ndarray
      3x1 reference Geodetic Latitude, Longitude, Height [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z coordinates [m]
  """
  C_e_n = ecef2nedDcm(lla0)
  xyz0 = lla2ecef(lla0)
  xyz = lla2ecef(lla)
  return C_e_n @ (xyz - xyz0)


# === LLA2ENU ===
@njit(cache=True, fastmath=True)
def lla2enu(lla: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Latitude-Longitude-Height to East-North-Up coordinates

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  lla0 : np.ndarray
      3x1 reference Geodetic Latitude, Longitude, Height [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ENU x,y,z coordinates [m]
  """
  C_e_n = ecef2enuDcm(lla0)
  xyz0 = lla2ecef(lla0)
  xyz = lla2ecef(lla)
  return C_e_n @ (xyz - xyz0)


# === LLA2AER ===
@njit(cache=True, fastmath=True)
def lla2aer(lla_t: np.ndarray, lla_r: np.ndarray) -> np.ndarray:
  """Converts Latitude-Longitude-Height to Azimuth-Elevation-Range coordinates

  Parameters
  ----------
  lla_t : np.ndarray
      3x1 target LLA coordinates
  lla_r : np.ndarray
      3x1 reference LLA coordinates

  Returns
  -------
  np.ndarray
      3x1 relative AER from reference to target
  """
  return enu2aer(lla2enu(lla_t, lla_r), ecef2enu(lla_r, lla_r))


#--------------------------------------------------------------------------------------------------#
# === ECI2ECEF ===
@njit(cache=True, fastmath=True)
def eci2ecef(xyz: np.ndarray, dt: np.float64) -> np.ndarray:
  """Earth-Centered-Inertial to Earth-Centered-Earth-Fixed coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 ECI x,y,z coordinates [m]
  dt : np.float64
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z coordinates [m]
  """
  C_i_e = eci2ecefDcm(dt)
  return C_i_e @ xyz


# === ECI2LLA ===
@njit(cache=True, fastmath=True)
def eci2lla(xyz: np.ndarray, dt: np.float64) -> np.ndarray:
  """Earth-Centered-Inertial to Latitude-Longitude-Height coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 ECI x,y,z coordinates [m]
  dt : np.float64
      time [s]

  Returns
  -------
  np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  """
  xyz = eci2ecef(xyz, dt)
  return ecef2lla(xyz)


# === ECI2NED ===
@njit(cache=True, fastmath=True)
def eci2ned(xyz: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
  """Earth-Centered-Inertial to North-East-Down coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 NED x,y,z coordinates [m]
  lla0 : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : np.float64
      time [s]

  Returns
  -------
  np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  """
  xyz = eci2ecef(xyz, dt)
  return ecef2ned(xyz, lla0)


# === ECI2ENU ===
@njit(cache=True, fastmath=True)
def eci2enu(xyz: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
  """Earth-Centered-Inertial to East-North-Up coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 ENU x,y,z coordinates [m]
  lla0 : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : np.float64
      time [s]

  Returns
  -------
  np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  """
  xyz = eci2ecef(xyz, dt)
  return ecef2enu(xyz, lla0)


# === ECI2AER ===
@njit(cache=True, fastmath=True)
def eci2aer(eci_t: np.ndarray, eci_r: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts Earth-Centered-Inertial to Azimuth-Elevation-Range coordinates

  Parameters
  ----------
  eci_t : np.ndarray
      3x1 target ECI coordinates
  eci_r : np.ndarray
      3x1 reference ECI coordinates
  dt : np.float64
      time [s]

  Returns
  -------
  np.ndarray
      3x1 relative AER from reference to target
  """
  lla0 = eci2lla(eci_r, dt)
  return enu2aer(eci2enu(eci_t, lla0, dt), eci2enu(eci_r, lla0, dt))


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECI ===
@njit(cache=True, fastmath=True)
def ecef2eci(xyz: np.ndarray, dt: np.float64) -> np.ndarray:
  """Earth-Centered-Earth-Fixed to Earth-Centered-Inertial coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 ECEF x,y,z coordinates [m]
  dt : np.float64
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECI x,y,z coordinates [m]
  """
  C_e_i = ecef2eciDcm(dt)
  return C_e_i @ xyz


# === ECEF2LLA ===
@njit(cache=True, fastmath=True)
def ecef2lla(xyz: np.ndarray) -> np.ndarray:
  """Earth-Centered-Earth-Fixed to Latitude-Longitude-Height coordinates
      - (Groves Appendix C) Borkowski closed form exact solution

  Parameters
  ----------
  xyz : np.ndarray
      3x1 ECEF x,y,z [m]

  Returns
  -------
  np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  """
  x,y,z = xyz
  
  beta = np.hypot(x, y)                                                           # (Groves C.18)
  a = np.sqrt(1 - WGS84_E2) * np.abs(z)
  b = WGS84_E2 * WGS84_R0
  E = (a - b) / beta                                                              # (Groves C.29)
  F = (a + b) / beta                                                              # (Groves C.30)
  P = 4/3 * (E*F + 1)                                                             # (Groves C.31)
  Q = 2 * (E*E - F*F)                                                             # (Groves C.32)
  D = P*P*P + Q*Q                                                                 # (Groves C.33)
  V = (np.sqrt(D) - Q)**(1/3) - (np.sqrt(D) + Q)**(1/3)                           # (Groves C.34)
  G = 0.5 * (np.sqrt(E*E + V) + E)                                                # (Groves C.35)
  T = np.sqrt( G*G + ((F - V*G) / (2*G - E)) ) - G                                # (Groves C.36)
  phi = np.sign(z) * np.arctan( (1 - T*T) / (2*T*np.sqrt(1 - WGS84_E2)) )               # (Groves C.37)
  h = (beta - WGS84_R0*T)*np.cos(phi) + (z - np.sign(z)*WGS84_R0*np.sqrt(1 - WGS84_E2))*np.sin(phi) # (Groves C.38)

  # combine lla
  lamb = np.arctan2(y, x)
  lla = np.array([phi, lamb, h], dtype=np.double)
  
  return np.array([phi, lamb, h], dtype=np.double)


# === ECEF2NED ===
@njit(cache=True, fastmath=True)
def ecef2ned(xyz: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Earth-Centered-Earth-Fixed to North-East-Down coordinates

  Parameters
  ----------
  xyz : np.ndarray
      ECEF x,y,z [m]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      NED x,y,z coordinates [m]
  """
  C_e_n = ecef2nedDcm(lla0)
  xyz0 = lla2ecef(lla0)
  return C_e_n @ (xyz - xyz0)


# === ECEF2ENU ===
@njit(cache=True, fastmath=True)
def ecef2enu(xyz: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Earth-Centered-Earth-Fixed to East-North-Up coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 ECEF x,y,z [m]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ENU x,y,z coordinates [m]
  """
  C_e_n = ecef2enuDcm(lla0)
  xyz0 = lla2ecef(lla0)
  return C_e_n @ (xyz - xyz0)


# === ECEF2AER ===
@njit(cache=True, fastmath=True)
def ecef2aer(ecef_t: np.ndarray, ecef_r: np.ndarray) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to Azimuth-Elevation-Range coordinates

  Parameters
  ----------
  ecef_t : np.ndarray
      3x1 target ECEF coordinates
  ecef_r : np.ndarray
      3x1 reference ECEF coordinates

  Returns
  -------
  np.ndarray
      3x1 relative AER from reference to target
  """
  lla0 = ecef2lla(ecef_r)
  return enu2aer(ecef2enu(ecef_t, lla0), ecef2enu(ecef_r, lla0))


#--------------------------------------------------------------------------------------------------#
# === NED2ECI ===
@njit(cache=True, fastmath=True)
def ned2eci(ned: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
  """North-East-Down to Earth-Centered-Inertial coordinates

  Parameters
  ----------
  ned : np.ndarray
      3x1 NED x,y,z [m]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : np.float64
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECI x,y,z coordinates [m]
  """
  xyz = ned2ecef(ned, lla0)
  C_e_i = ecef2eciDcm(dt)
  return C_e_i @ xyz


# === NED2ECEF ===
@njit(cache=True, fastmath=True)
def ned2ecef(ned: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """North-East-Down to Earth-Centered-Earth-Fixed coordinates

  Parameters
  ----------
  ned : np.ndarray
      3x1 NED x,y,z [m]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ECI x,y,z coordinates [m]
  """
  C_n_e = ned2ecefDcm(lla0)
  xyz = lla2ecef(lla0)
  return xyz + C_n_e @ ned
  

# === NED2LLA ===
@njit(cache=True, fastmath=True)
def ned2lla(ned: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """North-East-Down to Latitude-Longitude-Height coordinates

  Parameters
  ----------
  ned : np.ndarray
      3x1 NED x,y,z [m]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  """
  xyz = ned2ecef(ned, lla0)
  return ecef2lla(xyz)


# === NED2AER ===
@njit(cache=True, fastmath=True)
def ned2aer(ned_t: np.ndarray, ned_r: np.ndarray) -> np.ndarray:
  """Converts North-East-Down to Azimuth-Elevation-Range coordinates

  Parameters
  ----------
  ned_t : np.ndarray
      3x1 target NED coordinates
  ned_r : np.ndarray
      3x1 reference NED coordinates

  Returns
  -------
  np.ndarray
      3x1 relative AER from reference to target
  """
  dn, de, dd = ned_t - ned_r

  r = np.hypot(de, dn)
  az = np.mod(np.arctan2(de, dn), 2*np.pi)
  el = np.arctan2(-dd, r)
  rng = np.hypot(r, -dd)
  
  return np.array([az, el, rng], dtype=np.double)


#--------------------------------------------------------------------------------------------------#
# === ENU2ECI ===
@njit(cache=True, fastmath=True)
def enu2eci(enu: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
  """East-North-Up to Earth-Centered-Inertial coordinates

  Parameters
  ----------
  enu : np.ndarray
      3x1 ENU x,y,z [m]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : np.float64
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECI x,y,z coordinates [m]
  """
  xyz = enu2ecef(enu, lla0)
  C_e_i = ecef2eciDcm(dt)
  return C_e_i @ xyz


# === ENU2ECEF ===
@njit(cache=True, fastmath=True)
def enu2ecef(enu: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """East-North-Up to Earth-Centered-Earth-Fixed coordinates

  Parameters
  ----------
  enu : np.ndarray
      3x1 ENU x,y,z [m]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ECI x,y,z coordinates [m]
  """
  C_n_e = enu2ecefDcm(lla0)
  xyz = lla2ecef(lla0)
  return xyz + C_n_e @ enu
  

# === ENU2LLA ===
@njit(cache=True, fastmath=True)
def enu2lla(enu: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """East-North-Up to Latitude-Longitude-Height coordinates

  Parameters
  ----------
  enu : np.ndarray
      3x1 ENU x,y,z [m]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  """
  xyz = enu2ecef(enu, lla0)
  return ecef2lla(xyz)


# === ENU2AER ===
@njit(cache=True, fastmath=True)
def enu2aer(enu_t: np.ndarray, enu_r: np.ndarray) -> np.ndarray:
  """Converts East-North-Up to Azimuth-Elevation-Range coordinates

  Parameters
  ----------
  enu_t : np.ndarray
      3x1 target ENU coordinates
  enu_r : np.ndarray
      3x1 reference ENU coordinates

  Returns
  -------
  np.ndarray
      3x1 relative AER from reference to target
  """
  de, dn, du = enu_t - enu_r

  r = np.hypot(de, dn)
  az = np.mod(np.arctan2(de, dn), 2*np.pi)
  el = np.arctan2(du, r)
  rng = np.hypot(r, du)
  
  return np.array([az, el, rng], dtype=np.double)
