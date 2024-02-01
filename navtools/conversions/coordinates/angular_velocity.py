'''
|====================================== angular_velocity.py =======================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/coordinate/angular_velocity.py                                               |
|  @brief    Common angular velocity coordinate frame transformations.                             |
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
from .position import *
from .velocity import *
from navtools.constants import GNSS_OMEGA_EARTH, WGS84_E2, WGS84_R0

omega_ie = np.array([0.0, 0.0, GNSS_OMEGA_EARTH], dtype=np.float64)


#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFW ===
@njit(cache=True, fastmath=True)
def eci2ecefw(w_ib_i: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts Earth-Centered-Inertial to Earth-Centered-Earth-Fixed angular velocity

  Parameters
  ----------
  w_ib_i : np.ndarray
      3x1 ECI x,y,z angular velocity [m/s]
  dt : np.float64
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
def eci2nedw(w_ib_i: np.ndarray, r_ib_i: np.ndarray, v_ib_i: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
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
  dt : np.float64
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
def eci2enuw(w_ib_i: np.ndarray, r_ib_i: np.ndarray, v_ib_i: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
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
  dt : np.float64
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
def ecef2eciw(w_eb_e: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to Earth-Centered-Inertial angular velocity

  Parameters
  ----------
  w_eb_e : np.ndarray
      3x1 ECEF x,y,z angular velocity [m/s]
  dt : np.float64
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
def ned2eciw(w_nb_n: np.ndarray, v_nb_n: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts North-East-Down to East-Centered-Inertial angular velocity

  Parameters
  ----------
  w_nb_n : np.ndarray
      3x1 NED x,y,z angular velocity [m/s]
  v_nb_n : np.ndarray
      3x1 NED x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : np.float64
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
def enu2eciw(w_nb_n: np.ndarray, v_nb_n: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts East-North-Up to East-Centered-Inertial angular velocity

  Parameters
  ----------
  w_nb_n : np.ndarray
      3x1 ENU x,y,z angular velocity [m/s]
  v_nb_n : np.ndarray
      3x1 ENU x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : np.float64
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


