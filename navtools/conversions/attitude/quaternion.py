'''
|======================================== quaternion.py ===========================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/attitude/quaternion.py                                                       |
|  @brief    Attitude conversion from quaternions. All rotations assume right-hand                 |
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

# === QUAT2EULER ===
@njit(cache=True, fastmath=True)
def quat2euler(q: np.ndarray) -> np.ndarray:
  """Converts quaternion to corresponding euler angles (roll-pitch-yaw)

  Parameters
  ----------
  q : np.ndarray
      4x1 quaternion

  Returns
  -------
  np.ndarray
      3x1 RPY euler angles [radians]
  """
  w, x, y, z = q
  e = np.array([np.arctan2(2*(w*x + y*z), (w*w - x*x - y*y + z*z)), \
                np.arcsin(-2*(-w*y + x*z)), \
                np.arctan2(2*(w*z + x*y), (w*w + x*x - y*y - z*z))], \
               dtype=np.double)
  return e


# === QUAT2DCM ===
@njit(cache=True, fastmath=True)
def quat2dcm(q: np.ndarray) -> np.ndarray:
  """Converts quaternion to corresponding 'XYZ' DCM

  Parameters
  ----------
  q : np.ndarray
      4x1 quaternion

  Returns
  -------
  np.ndarray
      3x3 'ZYX' direction cosine matrix
  """
  w, x, y, z = q
  C = np.array([[w*w + x*x - y*y - z*z,         2*(x*y - w*z),          2*(w*y + x*z)], \
                [        2*(w*z + x*y), w*w - x*x + y*y - z*z,          2*(y*z - w*x)], \
                [        2*(x*z - w*y),         2*(y*z + w*x),  w*w - x*x - y*y + z*z]], \
               dtype=np.double)
  return C
