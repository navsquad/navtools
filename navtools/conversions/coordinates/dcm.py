'''
|============================================ dcm.py ==============================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/coordinate/dcm.py                                                            |
|  @brief    Common coordinate frame transformation matrices.                                      |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from navtools.constants import GNSS_OMEGA_EARTH

#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFDCM ===
@njit(cache=True, fastmath=True)
def eci2ecefDcm(dt: np.float64) -> np.ndarray:
  """Earth-Centered-Inertial to Earth-Centered-Earth-Fixed direction cosine matrix

  Parameters
  ----------
  dt : np.float64
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
def eci2nedDcm(lla: np.ndarray, dt: np.float64) -> np.ndarray:
  """Earth-Centered-Inertial to North-East-Down direction cosine matrix

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : np.float64
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
def eci2enuDcm(lla: np.ndarray, dt: np.float64) -> np.ndarray:
  """Earth-Centered-Inertial to East-North-Up direction cosine matrix

  Parameters
  ----------
  lla : np.ndarray
      3x1 Geodetic Latitude, Longitude, Height [rad, rad, m]
  dt : np.float64
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
def ecef2eciDcm(dt: np.float64) -> np.ndarray:
  """Earth-Centered-Earth-Fixed to Earth-Centered-Inertial direction cosine matrix

  Parameters
  ----------
  dt : np.float64
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
