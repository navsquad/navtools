'''
|======================================== acceleration.py =========================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/coordinate/acceleration.py                                                   |
|  @brief    Common acceleration coordinate frame transformations.                                 |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2023                                                                          |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from .dcm import *
from .position import *
from .velocity import *
from navtools.constants import GNSS_OMEGA_EARTH
from navtools.conversions.attitude.skew import skew

OMEGA_ie = skew(np.array([0.0, 0.0, GNSS_OMEGA_EARTH], dtype=np.float64))


#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFA ===
@njit(cache=True, fastmath=True)
def eci2ecefa(r_ib_i: np.ndarray, v_ib_i: np.ndarray, a_ib_i: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts Earth-Centered-Inertial to Earth-Centered-Earth-Fixed acceleration

  Parameters
  ----------
  r_ib_i : np.ndarray
      3x1 ECI x,y,z position [m]
  v_ib_i : np.ndarray
      3x1 ECI x,y,z velocity [m/s]
  a_ib_i : np.ndarray
      3x1 ECI x,y,z acceleration [m/s]
  dt : np.float64
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
def eci2neda(r_ib_i: np.ndarray, v_ib_i: np.ndarray, a_ib_i: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
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
  dt : np.float64
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
def eci2enua(r_ib_i: np.ndarray, v_ib_i: np.ndarray, a_ib_i: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
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
  dt : np.float64
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
def ecef2ecia(r_eb_e: np.ndarray, v_eb_e: np.ndarray, a_eb_e: np.ndarray, dt: np.float64) -> np.ndarray:
  """Converts Earth-Centered-Earth-Fixed to Earth-Centered-Inertial acceleration

  Parameters
  ----------
  r_eb_e : np.ndarray
      3x1 ECEF x,y,z position [m]
  v_eb_e : np.ndarray
      3x1 ECEF x,y,z velocity [m/s]
  a_eb_e : np.ndarray
      3x1 ECEF x,y,z acceleration [m/s]
  dt : np.float64
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
def ned2ecia(r_nb_n: np.ndarray, v_nb_n: np.ndarray, a_nb_n: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
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
  dt : np.float64
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
def enu2ecia(r_nb_n: np.ndarray, v_nb_n: np.ndarray, a_nb_n: np.ndarray, lla0: np.ndarray, dt: np.float64) -> np.ndarray:
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
  dt : np.float64
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
