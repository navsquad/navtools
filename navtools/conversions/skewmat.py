"""
|=========================================== skew.py ==============================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/attitude.py                                                                  |
|  @brief    Skew-symmetric forms of vectors.                                                      |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
"""

__all__ = ["skew", "deskew"]

import numpy as np
from numba import njit


# * ============================================================================================== *#
# === SKEW ===
@njit(cache=True, fastmath=True)
def skew(v: np.ndarray) -> np.ndarray:
    """Converts vector into its skew symmetric form

    Parameters
    ----------
    v : np.ndarray
        3x1 vector

    Returns
    -------
    np.ndarray
        3x3 skew symmetric form of vector
    """
    M = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.double
    )
    return M


# === DESKEW ===
@njit(cache=True, fastmath=True)
def deskew(M: np.ndarray) -> np.ndarray:
    """Converts skew symmetric form into its respective vector

    Parameters
    ----------
    M : np.ndarray
        3x3 skew symmetric form of vector

    Returns
    -------
    np.ndarray
        3x1 vector
    """
    v = np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.double)
    return v
