'''
|============================================ dcm.py ==============================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/attitude/dcm.py                                                              |
|  @brief    Attitude conversion from direction cosine matrices. All rotations assume right-hand   |
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
from .euler import euler2quat

# === DCM2EULER === 
@njit(cache=True, fastmath=True)
def dcm2euler(C: np.ndarray) -> np.ndarray:
  """Converts 'ZYX' DCM matrix into corresponding euler angles (roll-pitch-yaw)

  Parameters
  ----------
  C : np.ndarray
      3x3 'ZYX' direction cosine matrix

  Returns
  -------
  np.ndarray
      3x1 euler angles roll, pitch, yaw [rad]
  """
  e = np.array([np.arctan2(C[1,2], C[2,2]), \
                np.arcsin(-C[0,2]), \
                np.arctan2(C[0,1], C[0,0])], 
               dtype=np.double)
  return e


# === DCM2QUAT === 
@njit(cache=True, fastmath=True)
def dcm2quat(C: np.ndarray) -> np.ndarray:
  """Converts 'ZYX' DCM matrix into corresponding quaternion

  Parameters
  ----------
  C : np.ndarray
      3x1 'ZYX' direction cosine matrix

  Returns
  -------
  np.ndarray
      4x1 quaternion
  """
  q = euler2quat(dcm2euler(C))
  return q
