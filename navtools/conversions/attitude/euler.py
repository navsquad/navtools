'''
|=========================================== euler.py =============================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/attitude/euler.py                                                            |
|  @brief    Attitude conversion from euler angles. All rotations assume right-hand                |
|            coordinate frames with the order. Assumes euler angles in the order 'roll-pitch-yaw'  |
|            and DCMs with the order of 'ZYX'.                                                     |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit

# === EULER2DCM ===
@njit(cache=True, fastmath=True)
def euler2dcm(e: np.ndarray) -> np.ndarray:
  """Converts euler angles (roll-pitch-yaw) to corresponding 'ZYX' DCM

  Parameters
  ----------
  e : np.ndarray
      3x1 RPY euler angles [radians]

  Returns
  -------
  np.ndarray
      3x3 'ZYX' direction cosine matrix
  """
  sinP, sinT, sinS = np.sin(e)
  cosP, cosT, cosS = np.cos(e)
  C = np.array([[cosT*cosS, cosT*sinS, -sinT], \
                [sinP*sinT*cosS - cosP*sinS, sinP*sinT*sinS + cosP*cosS, cosT*sinP], \
                [sinT*cosP*cosS + sinS*sinP, sinT*cosP*sinS - cosS*sinP, cosT*cosP]],
               dtype=np.double)
  return C


# === EULER2QUAT ===
@njit(cache=True, fastmath=True)
def euler2quat(e: np.ndarray) -> np.ndarray:
  """Converts euler angles (roll-pitch-yaw) to corresponding quaternion

  Parameters
  ----------
  e : np.ndarray
      3x1 RPY euler angles [radians]

  Returns
  -------
  np.ndarray
      4x1 quaternion
  """
  sinX, sinY, sinZ = np.sin(e)
  cosX, cosY, cosZ = np.cos(e)
  q = np.array([[cosZ*cosY*cosX + sinZ*sinY*sinX], \
                [cosZ*cosY*sinX - sinZ*sinY*cosX], \
                [cosZ*sinY*cosX + sinZ*cosY*sinX], \
                [sinZ*cosY*cosX - cosZ*sinY*sinX]], 
               dtype=np.double)
  return q


# === ROT_X ===
@njit(cache=True, fastmath=True)
def rot_x(phi: np.float64) -> np.ndarray:
  """Converts single euler angle to corresponding 'X' DCM

  Parameters
  ----------
  phi : np.float64
      euler angle [radians]

  Returns
  -------
  np.ndarray
      3x3 direction cosine matrix
  """
  sinP = np.sin(phi)
  cosP = np.cos(phi)
  R = np.array([[1.0,  0.0,   0.0], \
                [0.0, cosP, -sinP], \
                [0.0, sinP,  cosP]], 
               dtype=np.double)
  return R


# === ROT_Y ===
@njit(cache=True, fastmath=True)
def rot_y(theta: np.float64) -> np.ndarray:
  """Converts single euler angle to corresponding 'Y' DCM

  Parameters
  ----------
  theta : np.float64
      euler angle [radians]

  Returns
  -------
  np.ndarray
      3x3 direction cosine matrix
  """
  sinT = np.sin(theta)
  cosT = np.cos(theta)
  R = np.array([[ cosT, 0.0, sinT], \
                [  0.0, 1.0,  0.0], \
                [-sinT, 0.0, cosT]], 
               dtype=np.double)
  return R


# === ROT_Z ===
@njit(cache=True, fastmath=True)
def rot_z(psi: np.float64) -> np.ndarray:
  """Converts single euler angle to corresponding 'Z' DCM

  Parameters
  ----------
  psi : np.float64
      euler angle [radians]

  Returns
  -------
  np.ndarray
      3x3 direction cosine matrix
  """
  sinS = np.sin(psi)
  cosS = np.cos(psi)
  R = np.array([[cosS, -sinS, 0.0], \
                [sinS,  cosS, 0.0], \
                [ 0.0,   0.0, 1.0]], 
               dtype=np.double)
  return R
