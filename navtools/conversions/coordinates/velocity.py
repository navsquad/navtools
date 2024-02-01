'''
|========================================== velcoity.py ===========================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/coordinate/velocity.py                                                       |
|  @brief    Common velocity coordinate frame transformations.                                     |
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
from navtools.constants import GNSS_OMEGA_EARTH
from navtools.conversions.attitude.skew import skew

OMEGA_ie = skew(np.array([0.0, 0.0, GNSS_OMEGA_EARTH], dtype=np.float64))

#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFV ===
@njit(cache=True, fastmath=True)
def eci2ecefv(r_ib_i: np.ndarray, v_ib_i: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts Earth-Centered-Inertial to Earth-Centered-Earth-Fixed velocity

  Parameters
  ----------
  r_ib_i : np.ndarray
      3x1 ECI x,y,z position [m]
  v_ib_i : np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  dt : np.float64
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
def eci2nedv(r_ib_i: np.ndarray, v_ib_i: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts Earth-Centered-Inertial to North-East-Down velocity

  Parameters
  ----------
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
      3x1 NED x,y,z velocity [m/s]
  """
  C_i_n = eci2nedDcm(lla0, dt)
  return C_i_n @ (v_ib_i - OMEGA_ie @ r_ib_i)


# === ECI2ENUV ===
@njit(cache=True, fastmath=True)
def eci2enuv(r_ib_i: np.ndarray, v_ib_i: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts Earth-Centered-Inertial to East-North-Up velocity

  Parameters
  ----------
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
      3x1 ENU x,y,z velocity [m/s]
  """
  C_i_n = eci2enuDcm(lla0, dt)
  return C_i_n @ (v_ib_i - OMEGA_ie @ r_ib_i)


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECIV ===
@njit(cache=True, fastmath=True)
def ecef2eciv(r_eb_e: np.ndarray, v_eb_e: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to Earth-Centered-Inertial velocity

  Parameters
  ----------
  r_eb_e : np.ndarray
      3x1 ECEF x,y,z position [m]
  v_eb_e : np.ndarray
      3x1 ECEF x,y,z velocity [m/s]
  dt : np.float64
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
def ned2eciv(ned: np.ndarray, v_ned: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts North-East-Down to East-Centered-Inertial velocity

  Parameters
  ----------
  ned : np.ndarray
      3x1 NED x,y,z position [m/s]
  v_ned : np.ndarray
      3x1 NED x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : np.float64
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
def enu2eciv(r_nb_n: np.ndarray, v_nb_n: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts East-North-Up to East-Centered-Inertial velocity

  Parameters
  ----------
  r_nb_n : np.ndarray
      3x1 ENU x,y,z position [m/s]
  v_nb_n : np.ndarray
      3x1 ENU x,y,z velocity [m/s]
  lla0 : np.ndarray
      3x1 LLA [rad, rad, m]
  dt : np.float64
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
