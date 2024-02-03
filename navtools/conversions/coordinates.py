'''
|======================================== coordinates.py ==========================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/coordinates.py                                                               |
|  @brief    Common coordinate frame transformations.                                              |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from navtools.constants import GNSS_OMEGA_EARTH, WGS84_R0, WGS84_E2
from navtools.conversions.skew import skew

omega_ie = np.array([0.0, 0.0, GNSS_OMEGA_EARTH], dtype=float)
OMEGA_ie = skew(np.array([0.0, 0.0, GNSS_OMEGA_EARTH], dtype=float))


#* ============================================================================================== *#
#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFDCM ===
@njit(cache=True, fastmath=True)
def eci2ecefDcm(dt: float) -> np.ndarray:
  """Earth-Centered-Inertial to Earth-Centered-Earth-Fixed direction cosine matrix

  Parameters
  ----------
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x3 ECI->ECEF direction cosine matrix
  """
  sin_wie = np.sin(GNSS_OMEGA_EARTH*dt)
  cos_wie = np.cos(GNSS_OMEGA_EARTH*dt)
  # Groves 2.145
  C_i_e = np.array([[ cos_wie, sin_wie, 0.0], \
                    [-sin_wie, cos_wie, 0.0], \
                    [     0.0,     0.0, 1.0]], \
                   dtype=np.double)
  return C_i_e


# === ECI2NEDDCM ===
@njit(cache=True, fastmath=True)
def eci2nedDcm(lla: np.ndarray, dt: float) -> np.ndarray:
  """Earth-Centered-Inertial to North-East-Down direction cosine matrix

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x3 ECI->NAV direction cosine matrix
  """
  sinPhi = np.sin(lla[0])
  cosPhi = np.cos(lla[0])
  sin_lam_wie = np.sin(lla[1] + GNSS_OMEGA_EARTH*dt)
  cos_lam_wie = np.cos(lla[1] + GNSS_OMEGA_EARTH*dt)
  # Groves 2.154
  C_i_n = np.array([[-sinPhi*cos_lam_wie, -sinPhi*sin_lam_wie,  cosPhi], \
                    [       -sin_lam_wie,         cos_lam_wie,     0.0], \
                    [-cosPhi*cos_lam_wie, -cosPhi*sin_lam_wie, -sinPhi]], 
                   dtype=np.double)
  return C_i_n


# === ECI2NEDDCM ===
@njit(cache=True, fastmath=True)
def eci2enuDcm(lla: np.ndarray, dt: float) -> np.ndarray:
  """Earth-Centered-Inertial to East-North-Up direction cosine matrix

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x3 ECI->NAV direction cosine matrix
  """
  sinPhi = np.sin(lla[0])
  cosPhi = np.cos(lla[0])
  sin_lam_wie = np.sin(lla[1] + GNSS_OMEGA_EARTH*dt)
  cos_lam_wie = np.cos(lla[1] + GNSS_OMEGA_EARTH*dt)
  C_i_n = np.array([[       -sin_lam_wie,         cos_lam_wie,    0.0], \
                    [-sinPhi*cos_lam_wie, -sinPhi*sin_lam_wie, cosPhi], \
                    [ cosPhi*cos_lam_wie,  cosPhi*sin_lam_wie, sinPhi]], 
                   dtype=np.double)
  return C_i_n


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECIDCM ===
@njit(cache=True, fastmath=True)
def ecef2eciDcm(dt: float) -> np.ndarray:
  """Earth-Centered-Earth-Fixed to Earth-Centered-Inertial direction cosine matrix

  Parameters
  ----------
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x3 ECEF->ECI direction cosine matrix
  """
  sin_wie = np.sin(GNSS_OMEGA_EARTH*dt)
  cos_wie = np.cos(GNSS_OMEGA_EARTH*dt)
  # Groves 2.145
  C_e_i = np.array([[cos_wie, -sin_wie, 0.0], \
                    [sin_wie,  cos_wie, 0.0], \
                    [    0.0,      0.0, 1.0]], \
                   dtype=np.double)
  return C_e_i


# === ECEF2NEDDCM ===
@njit(cache=True, fastmath=True)
def ecef2nedDcm(lla: np.ndarray) -> np.ndarray:
  """Earth-Centered-Earth-Fixed to North-East-Down direction cosine matrix

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x3 ECEF->NAV direction cosine matrix
  """
  sinPhi, sinLam = np.sin(lla[:2])
  cosPhi, cosLam = np.cos(lla[:2])
  # Groves 2.150
  C_e_n = np.array([[-sinPhi*cosLam, -sinPhi*sinLam,  cosPhi], \
                    [       -sinLam,         cosLam,     0.0], \
                    [-cosPhi*cosLam, -cosPhi*sinLam, -sinPhi]], \
                   dtype=np.double)
  return C_e_n


# === ECEF2ENUDCM ===
@njit(cache=True, fastmath=True)
def ecef2enuDcm(lla: np.ndarray) -> np.ndarray:
  """Earth-Centered-Earth-Fixed to East-North-Up direction cosine matrix

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x3 ECEF->NAV direction cosine matrix
  """
  sinPhi, sinLam = np.sin(lla[:2])
  cosPhi, cosLam = np.cos(lla[:2])
  C_e_n = np.array([[       -sinLam,         cosLam,    0.0], \
                    [-sinPhi*cosLam, -sinPhi*sinLam, cosPhi], \
                    [ cosPhi*cosLam,  cosPhi*sinLam, sinPhi]], \
                   dtype=np.double)
  return C_e_n


#--------------------------------------------------------------------------------------------------#
# === NED2ECIDCM ===
@njit(cache=True, fastmath=True)
def ned2eciDcm(lla: np.ndarray, dt: np.ndarray) -> np.ndarray:
  """North-East-Down to Earth-Centered-Inertial direction cosine matrix

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : np.ndarray
      time [s]

  Returns
  -------
  np.ndarray
      3x3 NAV->ECI direction cosine matrix
  """
  sinPhi = np.sin(lla[0])
  cosPhi = np.cos(lla[0])
  sin_lam_wie = np.sin(lla[1] + GNSS_OMEGA_EARTH*dt)
  cos_lam_wie = np.cos(lla[1] + GNSS_OMEGA_EARTH*dt)
  # Groves 2.154
  C_n_i = np.array([[-sinPhi*cos_lam_wie, -sin_lam_wie, -cosPhi*cos_lam_wie], \
                    [-sinPhi*sin_lam_wie,  cos_lam_wie, -cosPhi*sin_lam_wie], \
                    [             cosPhi,          0.0,             -sinPhi]], \
                   dtype=np.double)
  return C_n_i


# === NED2ECEFDCM ===
@njit(cache=True, fastmath=True)
def ned2ecefDcm(lla: np.ndarray) -> np.ndarray:
  """North-East-Down to Earth-Centered-Earth-Fixed direction cosine matrix

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x3 NAV->ECEF direction cosine matrix
  """
  sinPhi, sinLam = np.sin(lla[:2])
  cosPhi, cosLam = np.cos(lla[:2])
  # Groves 2.150
  C_n_e = np.array([[-sinPhi*cosLam, -sinLam, -cosPhi*cosLam], \
                    [-sinPhi*sinLam,  cosLam, -cosPhi*sinLam], \
                    [        cosPhi,     0.0,        -sinPhi]], \
                   dtype=np.double)
  return C_n_e


#--------------------------------------------------------------------------------------------------#
# === ENU2ECIDCM ===
@njit(cache=True, fastmath=True)
def enu2eciDcm(lla: np.ndarray, dt: np.ndarray) -> np.ndarray:
  """East-North-Up to Earth-Centered-Inertial direction cosine matrix

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : np.ndarray
      time [s]

  Returns
  -------
  np.ndarray
      3x3 NAV->ECI direction cosine matrix
  """
  sinPhi = np.sin(lla[0])
  cosPhi = np.cos(lla[0])
  sin_lam_wie = np.sin(lla[1] + GNSS_OMEGA_EARTH*dt)
  cos_lam_wie = np.cos(lla[1] + GNSS_OMEGA_EARTH*dt)
  # Groves 2.154
  C_n_i = np.array([[-sin_lam_wie, -sinPhi*cos_lam_wie, cosPhi*cos_lam_wie], \
                    [ cos_lam_wie, -sinPhi*sin_lam_wie, cosPhi*sin_lam_wie], \
                    [         0.0,              cosPhi,             sinPhi]], \
                   dtype=np.double)
  return C_n_i


# === ENU2ECEFDCM ===
@njit(cache=True, fastmath=True)
def enu2ecefDcm(lla: np.ndarray) -> np.ndarray:
  """East-North-Up to Earth-Centered-Earth-Fixed direction cosine matrix

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x3 NAV->ECEF direction cosine matrix
  """
  sinPhi, sinLam = np.sin(lla[:2])
  cosPhi, cosLam = np.cos(lla[:2])
  C_n_e = np.array([[-sinLam, -cosLam*sinPhi, cosLam*cosPhi], \
                    [ cosLam, -sinLam*sinPhi, sinLam*cosPhi], \
                    [    0.0,         cosPhi,        sinPhi]], \
                   dtype=np.double)
  return C_n_e


#* ============================================================================================== *#
#--------------------------------------------------------------------------------------------------#
# === LLA2ECI ===
@njit(cache=True, fastmath=True)
def lla2eci(lla: np.ndarray, dt: float) -> np.ndarray:
  """Latitude-Longitude-Height to Earth-Centered-Inertial coordinates

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : float
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
  enu = lla2enu(lla_t, lla_r)
  r = np.linalg.norm(enu)
  az = np.arctan2(enu[0], enu[1])
  el = np.arcsin(enu[2], r)
  return np.array([az, el, r], dtype=np.double)


#--------------------------------------------------------------------------------------------------#
# === ECI2ECEF ===
@njit(cache=True, fastmath=True)
def eci2ecef(xyz: np.ndarray, dt: float) -> np.ndarray:
  """Earth-Centered-Inertial to Earth-Centered-Earth-Fixed coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 ECI x,y,z coordinates [m]
  dt : float
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
def eci2lla(xyz: np.ndarray, dt: float) -> np.ndarray:
  """Earth-Centered-Inertial to Latitude-Longitude-Height coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 ECI x,y,z coordinates [m]
  dt : float
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
def eci2ned(xyz: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Earth-Centered-Inertial to North-East-Down coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 NED x,y,z coordinates [m]
  lla0 : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : float
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
def eci2enu(xyz: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Earth-Centered-Inertial to East-North-Up coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 ENU x,y,z coordinates [m]
  lla0 : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : float
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
def eci2aer(eci_t: np.ndarray, eci_r: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Inertial to Azimuth-Elevation-Range coordinates

  Parameters
  ----------
  eci_t : np.ndarray
      3x1 target ECI coordinates
  eci_r : np.ndarray
      3x1 reference ECI coordinates
  dt : float
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
def ecef2eci(xyz: np.ndarray, dt: float) -> np.ndarray:
  """Earth-Centered-Earth-Fixed to Earth-Centered-Inertial coordinates

  Parameters
  ----------
  xyz : np.ndarray
      3x1 ECEF x,y,z coordinates [m]
  dt : float
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
  phi = np.sign(z) * np.arctan( (1 - T*T) / (2*T*np.sqrt(1 - WGS84_E2)) )         # (Groves C.37)
  h = (beta - WGS84_R0*T)*np.cos(phi) + (z - np.sign(z)*WGS84_R0*                 # (Groves C.38)
        np.sqrt(1 - WGS84_E2))*np.sin(phi) 

  # combine lla
  lamb = np.arctan2(y, x)
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
  enu = ecef2enu(ecef_t, lla0)
  r = np.linalg.norm(enu)
  az = np.arctan2(enu[0], enu[1])
  el = np.arcsin(enu[2], r)
  return np.array([az, el, r], dtype=np.double)
#   return enu2aer(ecef2enu(ecef_t, lla0), ecef2enu(ecef_r, lla0))


#--------------------------------------------------------------------------------------------------#
# === NED2ECI ===
@njit(cache=True, fastmath=True)
def ned2eci(ned: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """North-East-Down to Earth-Centered-Inertial coordinates

  Parameters
  ----------
  ned : np.ndarray
      3x1 NED x,y,z [m]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
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
def enu2eci(enu: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """East-North-Up to Earth-Centered-Inertial coordinates

  Parameters
  ----------
  enu : np.ndarray
      3x1 ENU x,y,z [m]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
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


#* ============================================================================================== *#
#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFV ===
@njit(cache=True, fastmath=True)
def eci2ecefv(r_ib_i: np.ndarray, v_ib_i: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Inertial to Earth-Centered-Earth-Fixed velocity

  Parameters
  ----------
  r_ib_i : np.ndarray
      3x1 ECI x,y,z position [m]
  v_ib_i : np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z velocity [m/s]
  """
  C_i_e = eci2ecefDcm(dt)
  return C_i_e @ (v_ib_i - OMEGA_ie @ r_ib_i)


# === ECI2NEDV ===
@njit(cache=True, fastmath=True)
def eci2nedv(r_ib_i: np.ndarray, v_ib_i: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Inertial to North-East-Down velocity

  Parameters
  ----------
  r_ib_i : np.ndarray
      3x1 ECI x,y,z position [m]
  v_ib_i : np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z velocity [m/s]
  """
  C_i_n = eci2nedDcm(lla0, dt)
  return C_i_n @ (v_ib_i - OMEGA_ie @ r_ib_i)


# === ECI2ENUV ===
@njit(cache=True, fastmath=True)
def eci2enuv(r_ib_i: np.ndarray, v_ib_i: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Inertial to East-North-Up velocity

  Parameters
  ----------
  r_ib_i : np.ndarray
      3x1 ECI x,y,z position [m]
  v_ib_i : np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ENU x,y,z velocity [m/s]
  """
  C_i_n = eci2enuDcm(lla0, dt)
  return C_i_n @ (v_ib_i - OMEGA_ie @ r_ib_i)


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECIV ===
@njit(cache=True, fastmath=True)
def ecef2eciv(r_eb_e: np.ndarray, v_eb_e: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to Earth-Centered-Inertial velocity

  Parameters
  ----------
  r_eb_e : np.ndarray
      3x1 ECEF x,y,z position [m]
  v_eb_e : np.ndarray
      3x1 ECEF x,y,z velocity [m/s]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z velocity [m/s]
  """
  C_e_i = ecef2eciDcm(dt)
  return C_e_i @ (v_eb_e - OMEGA_ie @ r_eb_e)


# === ECEF2NEDV ===
@njit(cache=True, fastmath=True)
def ecef2nedv(v_eb_e: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to North-East-Down velocity

  Parameters
  ----------
  v_eb_e : np.ndarray
      3x1 ECEF x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z velocity [m/s]
  """
  C_e_n = ecef2nedDcm(lla0)
  return C_e_n @ v_eb_e


# === ECEF2ENUV ===
@njit(cache=True, fastmath=True)
def ecef2enuv(v_eb_e: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to East-North-Up velocity

  Parameters
  ----------
  v_eb_e : np.ndarray
      3x1 ECEF x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z velocity [m/s]
  """
  C_e_n = ecef2enuDcm(lla0)
  return C_e_n @ v_eb_e


#--------------------------------------------------------------------------------------------------#
# === NED2ECIV ===
@njit(cache=True, fastmath=True)
def ned2eciv(ned: np.ndarray, v_ned: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts North-East-Down to East-Centered-Inertial velocity

  Parameters
  ----------
  ned : np.ndarray
      3x1 NED x,y,z position [m/s]
  v_ned : np.ndarray
      3x1 NED x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  """
  C_n_i = ned2eciDcm(lla0, dt)
  C_e_i = ecef2eciDcm(dt)
  xyz = ned2ecef(ned, lla0)
  return C_n_i @ v_ned + C_e_i @ OMEGA_ie @ xyz


# === NED2ECEFV ===
@njit(cache=True, fastmath=True)
def ned2ecefv(v_ned: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts North-East-Down to East-Centered-Earth-Fixed velocity

  Parameters
  ----------
  v_ned : np.ndarray
      3x1 NED x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z velocity [m/s]
  """
  C_n_e = ned2ecefDcm(lla0)
  return C_n_e @ v_ned


#--------------------------------------------------------------------------------------------------#
# === ENU2ECIV ===
@njit(cache=True, fastmath=True)
def enu2eciv(r_nb_n: np.ndarray, v_nb_n: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts East-North-Up to East-Centered-Inertial velocity

  Parameters
  ----------
  r_nb_n : np.ndarray
      3x1 ENU x,y,z position [m/s]
  v_nb_n : np.ndarray
      3x1 ENU x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  """
  C_n_i = enu2eciDcm(lla0, dt)
  C_e_i = ecef2eciDcm(dt)
  r_eb_e = enu2ecef(r_nb_n, lla0)
  return C_n_i @ v_nb_n + C_e_i @ OMEGA_ie @ r_eb_e


# === ENU2ECEFV ===
@njit(cache=True, fastmath=True)
def enu2ecefv(v_nb_n: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts East-North-Up to East-Centered-Earth-Fixed velocity

  Parameters
  ----------
  v_nb_n : np.ndarray
      3x1 ENU x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z velocity [m/s]
  """
  C_n_e = enu2ecefDcm(lla0)
  return C_n_e @ v_nb_n


#* ============================================================================================== *#
#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFW ===
@njit(cache=True, fastmath=True)
def eci2ecefw(w_ib_i: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Inertial to Earth-Centered-Earth-Fixed angular velocity

  Parameters
  ----------
  w_ib_i : np.ndarray
      3x1 ECI x,y,z angular velocity [m/s]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      ECEF x,y,z angular velocity [m/s]
  """
  C_i_e = eci2ecefDcm(dt)
  return C_i_e @ (w_ib_i + omega_ie)


# === ECI2NEDW ===
@njit(cache=True, fastmath=True)
def eci2nedw(w_ib_i: np.ndarray, r_ib_i: np.ndarray, v_ib_i: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Inertial to North-East-Down angular velocity

  Parameters
  ----------
  w_ib_i : np.ndarray
      3x1 ECI x,y,z angular velocity [m/s]
  r_ib_i : np.ndarray
      3x1 ECI x,y,z position [m]
  v_ib_i : np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z angular velocity [m/s]
  """
  C_i_n = eci2nedDcm(lla0, dt)
  
  vn, ve, vd = eci2nedv(r_ib_i, v_ib_i, lla0, dt)
  phi, lam, h = lla0
  sinPhi = np.sin(phi)
  
  trans = 1 - WGS84_E2 * sinPhi*sinPhi
  Re = WGS84_R0 / np.sqrt(trans)
  Rn = WGS84_R0 * (1- WGS84_E2) / trans**1.5
  w_en_n = np.array([ ve / (Re + h), \
                     -vn / (Rn + h), \
                     -ve * np.tan(phi) / (Re + h)], 
                    dtype=np.double)
  
  return C_i_n @ (w_ib_i - omega_ie) - w_en_n


# === ECI2ENUW ===
@njit(cache=True, fastmath=True)
def eci2enuw(w_ib_i: np.ndarray, r_ib_i: np.ndarray, v_ib_i: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Inertial to East-North-Up angular velocity

  Parameters
  ----------
  w_ib_i : np.ndarray
      3x1 ECI x,y,z angular velocity [m/s]
  r_ib_i : np.ndarray
      3x1 ECI x,y,z position [m]
  v_ib_i : np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z angular velocity [m/s]
  """
  C_i_n = eci2enuDcm(lla0, dt)
  
  ve, vn, vu = eci2enuv(r_ib_i, v_ib_i, lla0, dt)
  phi, lam, h = lla0
  sinPhi = np.sin(phi)
  
  trans = 1 - WGS84_E2 * sinPhi*sinPhi
  Re = WGS84_R0 / np.sqrt(trans)
  Rn = WGS84_R0 * (1- WGS84_E2) / trans**1.5
  w_en_n = np.array([ ve / (Re + h), \
                     -vn / (Rn + h), \
                     -ve * np.tan(phi) / (Re + h)], 
                    dtype=np.double)
  
  return C_i_n @ (w_ib_i - omega_ie) - w_en_n


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECIW ===
@njit(cache=True, fastmath=True)
def ecef2eciw(w_eb_e: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to Earth-Centered-Inertial angular velocity

  Parameters
  ----------
  w_eb_e : np.ndarray
      3x1 ECEF x,y,z angular velocity [m/s]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z angular velocity [m/s]
  """
  C_e_i = ecef2eciDcm(dt)
  return C_e_i @ (w_eb_e + omega_ie)

# === ECEF2NED ===
@njit(cache=True, fastmath=True)
def ecef2nedw(w_eb_e: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to North-East-Down angular velocity

  Parameters
  ----------
  w_eb_e : np.ndarray
      3x1 ECEF x,y,z angular velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z angular velocity [m/s]
  """
  C_e_n = ecef2nedDcm(lla0)
  return C_e_n @ w_eb_e


# === ECEF2NEDW ===
@njit(cache=True, fastmath=True)
def ecef2enuw(w_eb_e: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to East-North-Up angular velocity

  Parameters
  ----------
  w_eb_e : np.ndarray
      3x1 ECEF x,y,z angular velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ENU x,y,z angular velocity [m/s]
  """
  C_e_n = ecef2enuDcm(lla0)
  return C_e_n @ w_eb_e


#--------------------------------------------------------------------------------------------------#
# === NED2ECIW ===
@njit(cache=True, fastmath=True)
def ned2eciw(w_nb_n: np.ndarray, v_nb_n: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts North-East-Down to East-Centered-Inertial angular velocity

  Parameters
  ----------
  w_nb_n : np.ndarray
      3x1 NED x,y,z angular velocity [m/s]
  v_nb_n : np.ndarray
      3x1 NED x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z angular velocity [m/s]
  """
  C_n_i = ned2eciDcm(lla0, dt)
  
  vn, ve, vd = v_nb_n
  phi, lam, h = lla0
  sinPhi = np.sin(phi)
  
  trans = 1 - WGS84_E2 * sinPhi*sinPhi
  Re = WGS84_R0 / np.sqrt(trans)
  Rn = WGS84_R0 * (1- WGS84_E2) / trans**1.5
  w_en_n = np.array([ ve / (Re + h), \
                     -vn / (Rn + h), \
                     -ve * np.tan(phi) / (Re + h)], 
                    dtype=np.double)
  
  return C_n_i @ (w_nb_n + w_en_n) + omega_ie


# === NED2ECEFW ===
@njit(cache=True, fastmath=True)
def ned2ecefww(w_eb_e: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts North-East-Down to East-Centered-Earth-Fixed angular velocity

  Parameters
  ----------
  w_eb_e : np.ndarray
      3x1 NED x,y,z angular velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z angular velocity [m/s]
  """
  C_n_e = ned2ecefDcm(lla0)
  return C_n_e @ w_eb_e


#--------------------------------------------------------------------------------------------------#
# === ENU2ECIW ===
@njit(cache=True, fastmath=True)
def enu2eciw(w_nb_n: np.ndarray, v_nb_n: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts East-North-Up to East-Centered-Inertial angular velocity

  Parameters
  ----------
  w_nb_n : np.ndarray
      3x1 ENU x,y,z angular velocity [m/s]
  v_nb_n : np.ndarray
      3x1 ENU x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECI x,y,z angular velocity [m/s]
  """
  C_n_i = enu2eciDcm(lla0, dt)
  
  ve, vn, vu = v_nb_n
  phi, lam, h = lla0
  sinPhi = np.sin(phi)
  
  trans = 1 - WGS84_E2 * sinPhi*sinPhi
  Re = WGS84_R0 / np.sqrt(trans)
  Rn = WGS84_R0 * (1- WGS84_E2) / trans**1.5
  w_en_n = np.array([ ve / (Re + h), \
                     -vn / (Rn + h), \
                     -ve * np.tan(phi) / (Re + h)], 
                    dtype=np.double)
  
  return C_n_i @ (w_nb_n + w_en_n) + omega_ie


# === ENU2ECEFW ===
@njit(cache=True, fastmath=True)
def enu2ecefw(w_eb_e: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts East-North-Up to East-Centered-Earth-Fixed angular velocity

  Parameters
  ----------
  w_eb_e : np.ndarray
      3x1 ENU x,y,z angular valocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z angular velocity [m/s]
  """
  C_n_e = enu2ecefDcm(lla0)
  return C_n_e @ w_eb_e


#* ============================================================================================== *#
#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFA ===
@njit(cache=True, fastmath=True)
def eci2ecefa(r_ib_i: np.ndarray, v_ib_i: np.ndarray, a_ib_i: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Inertial to Earth-Centered-Earth-Fixed acceleration

  Parameters
  ----------
  r_ib_i : np.ndarray
      3x1 ECI x,y,z position [m]
  v_ib_i : np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  a_ib_i : np.ndarray
      3x1 ECI x,y,z acceleration [m/s]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z acceleration [m/s]
  """
  C_i_e = eci2ecefDcm(dt)
  return C_i_e @ (a_ib_i - 2 @ OMEGA_ie @ v_ib_i + OMEGA_ie @ OMEGA_ie @ r_ib_i)


# === ECI2NEDA ===
@njit(cache=True, fastmath=True)
def eci2neda(r_ib_i: np.ndarray, v_ib_i: np.ndarray, a_ib_i: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Inertial to North-East-Down acceleration

  Parameters
  ----------
  r_ib_i : np.ndarray
      3x1 ECI x,y,z position [m]
  v_ib_i : np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  a_ib_i : np.ndarray
      3x1 ECI x,y,z acceleration [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z acceleration [m/s]
  """
  C_i_n = eci2nedDcm(lla0, dt)
  return C_i_n @ (a_ib_i + 2 @ OMEGA_ie @ v_ib_i + OMEGA_ie @ OMEGA_ie @ r_ib_i)


# === ECI2ENUA ===
@njit(cache=True, fastmath=True)
def eci2enua(r_ib_i: np.ndarray, v_ib_i: np.ndarray, a_ib_i: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Inertial to East-North-Up acceleration

  Parameters
  ----------
  r_ib_i : np.ndarray
      3x1 ECI x,y,z position [m]
  v_ib_i : np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  a_ib_i : np.ndarray
      3x1 ECI x,y,z acceleration [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z acceleration [m/s]
  """
  C_i_n = eci2enuDcm(lla0, dt)
  return C_i_n @ (a_ib_i + 2 @ OMEGA_ie @ v_ib_i + OMEGA_ie @ OMEGA_ie @ r_ib_i)


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECIA ===
@njit(cache=True, fastmath=True)
def ecef2ecia(r_eb_e: np.ndarray, v_eb_e: np.ndarray, a_eb_e: np.ndarray, dt: float) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to Earth-Centered-Inertial acceleration

  Parameters
  ----------
  r_eb_e : np.ndarray
      3x1 ECEF x,y,z position [m]
  v_eb_e : np.ndarray
      3x1 ECEF x,y,z velocity [m/s]
  a_eb_e : np.ndarray
      3x1 ECEF x,y,z acceleration [m/s]
  dt : float
      time [s]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z acceleration [m/s]
  """
  C_i_e = eci2ecefDcm(dt)
  return C_i_e @ (a_eb_e - 2 @ OMEGA_ie @ v_eb_e + OMEGA_ie @ OMEGA_ie @ r_eb_e)


# === ECEF2NEDA ===
@njit(cache=True, fastmath=True)
def ecef2neda(a_eb_e: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to North-East-Down acceleration

  Parameters
  ----------
  a_eb_e : np.ndarray
      3x1 ECEF x,y,z acceleration [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z acceleration [m/s]
  """
  C_e_n = ecef2nedDcm(lla0)
  return C_e_n @ a_eb_e


# === ECEF2ENUA ===
@njit(cache=True, fastmath=True)
def ecef2neda(a_eb_e: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to East-North-Up acceleration

  Parameters
  ----------
  a_eb_e : np.ndarray
      3x1 ECEF x,y,z acceleration [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z acceleration [m/s]
  """
  C_e_n = ecef2enuDcm(lla0)
  return C_e_n @ a_eb_e


#--------------------------------------------------------------------------------------------------#
# === NED2ECIA ===
@njit(cache=True, fastmath=True)
def ned2ecia(r_nb_n: np.ndarray, v_nb_n: np.ndarray, a_nb_n: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts North-East-Down to East-Centered-Inertial acceleration

  Parameters
  ----------
  r_nb_n : np.ndarray
      3x1 NED x,y,z position [m/s]
  v_nb_n : np.ndarray
      3x1 NED x,y,z velocity [m/s]
  a_nb_n : np.ndarray
      3x1 NED x,y,z acceleration [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      3x1 time [s]

  Returns
  -------
  np.ndarray
      3x1 NED x,y,z acceleration [m/s]
  """
  r_eb_e = ned2ecef(r_nb_n, lla0)
  C_n_e = ned2ecefDcm(lla0)
  OMEGA_ie_n = skew(np.array([GNSS_OMEGA_EARTH*np.cos(lla0[0]), 0.0, -GNSS_OMEGA_EARTH*np.sin(lla0[0])]))
  C_n_i = ned2eciDcm(lla0, dt)
  C_e_i = ecef2eciDcm(lla0, dt)
  a_eb_n = C_n_e @ a_nb_n
  v_eb_n = C_n_e @ v_nb_n
  return C_n_i @ (a_eb_n + 2 @ OMEGA_ie_n @ v_eb_n) + C_e_i @ OMEGA_ie @ OMEGA_ie @ r_eb_e


# === NED2ECEFA ===
@njit(cache=True, fastmath=True)
def ned2ecefa(a_nb_n: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts North-East-Down to East-Centered-Earth-Fixed acceleration

  Parameters
  ----------
  a_nb_n : np.ndarray
      3x1 NED x,y,z acceleration [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z acceleration [m/s]
  """
  C_n_e = ned2ecefDcm(lla0)
  return C_n_e @ a_nb_n


#--------------------------------------------------------------------------------------------------#
# === ENU2ECIA ===
@njit(cache=True, fastmath=True)
def enu2ecia(r_nb_n: np.ndarray, v_nb_n: np.ndarray, a_nb_n: np.ndarray, lla0: np.ndarray, dt: float) -> np.ndarray:
  """Converts East-North-Up to East-Centered-Inertial acceleration

  Parameters
  ----------
  r_nb_n : np.ndarray
      3x1 ENU x,y,z position [m/s]
  v_nb_n : np.ndarray
      3x1 ENU x,y,z velocity [m/s]
  a_nb_n : np.ndarray
      3x1 ENU x,y,z acceleration [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : float
      3x1 time [s]

  Returns
  -------
  np.ndarray
      3x1 ECI x,y,z acceleration [m/s]
  """
  r_eb_e = enu2ecef(r_nb_n, lla0)
  C_n_e = enu2ecefDcm(lla0)
  OMEGA_ie_n = skew(np.array([GNSS_OMEGA_EARTH*np.cos(lla0[0]), 0.0, -GNSS_OMEGA_EARTH*np.sin(lla0[0])]))
  C_n_i = enu2eciDcm(lla0, dt)
  C_e_i = ecef2eciDcm(lla0, dt)
  a_eb_n = C_n_e @ a_nb_n
  v_eb_n = C_n_e @ v_nb_n
  return C_n_i @ (a_eb_n + 2 @ OMEGA_ie_n @ v_eb_n) + C_e_i @ OMEGA_ie @ OMEGA_ie @ r_eb_e


# === ENU2ECEFA ===
@njit(cache=True, fastmath=True)
def enu2ecefa(a_nb_n: np.ndarray, lla0: np.ndarray) -> np.ndarray:
  """Converts East-North-Up to East-Centered-Earth-Fixed acceleration

  Parameters
  ----------
  a_nb_n : np.ndarray
      3x1 ENU x,y,z acceleration [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]

  Returns
  -------
  np.ndarray
      3x1 ECEF x,y,z acceleration [m/s]
  """
  C_n_e = enu2ecefDcm(lla0)
  return C_n_e @ a_nb_n
