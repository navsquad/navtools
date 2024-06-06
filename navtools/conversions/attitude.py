"""
|========================================= attitude.py ============================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/attitude.py                                                                  |
|  @brief    Attitude conversion from direction cosine matrices. All rotations assume right-hand   |
|            coordinate frames with the order. Assumes euler angles in the order 'roll-pitch-yaw'  |
|            and DCMs with the order of 'ZYX'.                                                     |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
"""

__all__ = [
    "euler2dcm",
    "euler2quat",
    "rot_x",
    "rot_y",
    "rot_z",
    "dcm2euler",
    "dcm2quat",
    "quat2euler",
    "quat2dcm",
    "wrapTo2Pi",
    "wrapToPi",
    "wrapEulerAngles",
]

import numpy as np
from numba import njit

two_pi = 2 * np.pi
half_pi = 0.5 * np.pi


# * ============================================================================================== *#
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
    C = np.array(
        [
            [cosT * cosS, cosT * sinS, -sinT],
            [
                sinP * sinT * cosS - cosP * sinS,
                sinP * sinT * sinS + cosP * cosS,
                cosT * sinP,
            ],
            [
                sinT * cosP * cosS + sinS * sinP,
                sinT * cosP * sinS - cosS * sinP,
                cosT * cosP,
            ],
        ],
        dtype=np.double,
    )
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
    q = np.array(
        [
            cosZ * cosY * cosX + sinZ * sinY * sinX,
            cosZ * cosY * sinX - sinZ * sinY * cosX,
            cosZ * sinY * cosX + sinZ * cosY * sinX,
            sinZ * cosY * cosX - cosZ * sinY * sinX,
        ],
        dtype=np.double,
    )
    return q


# === ROT_X ===
@njit(cache=True, fastmath=True)
def rot_x(phi: float) -> np.ndarray:
    """Converts single euler angle to corresponding 'X' DCM

    Parameters
    ----------
    phi : float
        euler angle [radians]

    Returns
    -------
    np.ndarray
        3x3 direction cosine matrix
    """
    sinP = np.sin(phi)
    cosP = np.cos(phi)
    R = np.array([[1.0, 0.0, 0.0], [0.0, cosP, -sinP], [0.0, sinP, cosP]], dtype=np.double)
    return R


# === ROT_Y ===
@njit(cache=True, fastmath=True)
def rot_y(theta: float) -> np.ndarray:
    """Converts single euler angle to corresponding 'Y' DCM

    Parameters
    ----------
    theta : float
        euler angle [radians]

    Returns
    -------
    np.ndarray
        3x3 direction cosine matrix
    """
    sinT = np.sin(theta)
    cosT = np.cos(theta)
    R = np.array([[cosT, 0.0, sinT], [0.0, 1.0, 0.0], [-sinT, 0.0, cosT]], dtype=np.double)
    return R


# === ROT_Z ===
@njit(cache=True, fastmath=True)
def rot_z(psi: float) -> np.ndarray:
    """Converts single euler angle to corresponding 'Z' DCM

    Parameters
    ----------
    psi : float
        euler angle [radians]

    Returns
    -------
    np.ndarray
        3x3 direction cosine matrix
    """
    sinS = np.sin(psi)
    cosS = np.cos(psi)
    R = np.array([[cosS, -sinS, 0.0], [sinS, cosS, 0.0], [0.0, 0.0, 1.0]], dtype=np.double)
    return R


# * ============================================================================================== *#
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
    e = np.array(
        [
            np.arctan2(C[1, 2], C[2, 2]),
            np.arcsin(-C[0, 2]),
            np.arctan2(C[0, 1], C[0, 0]),
        ],
        dtype=np.double,
    )
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
    # q = euler2quat(dcm2euler(C))
    q_w = 0.5 * np.sqrt(1 + np.trace(C))
    if q_w > 0.01:
        q_w_4 = 4 * q_w
        q_x = (C[2, 1] - C[1, 2]) / q_w_4
        q_y = (C[0, 2] - C[2, 0]) / q_w_4
        q_z = (C[1, 0] - C[0, 1]) / q_w_4
    else:
        q_x = 0.5 * np.sqrt(1 + C[0, 0] - C[1, 1] - C[2, 2])
        q_x_4 = 4 * q_x
        q_w = (C[2, 1] - C[1, 2]) / q_x_4
        q_y = (C[0, 1] + C[1, 0]) / q_x_4
        q_z = (C[0, 2] + C[2, 0]) / q_x_4

    return np.array([q_w, q_x, q_y, q_z])


# * ============================================================================================== *#
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
    e = np.array(
        [
            np.arctan2(2 * (w * x + y * z), (w * w - x * x - y * y + z * z)),
            np.arcsin(-2 * (-w * y + x * z)),
            np.arctan2(2 * (w * z + x * y), (w * w + x * x - y * y - z * z)),
        ],
        dtype=np.double,
    )
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
    C = np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (w * y + x * z)],
            [2 * (w * z + x * y), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=np.double,
    )
    return C


# * ============================================================================================== *#
# === WRAPTO2PI ===
@njit(cache=True, fastmath=True)
def wrapTo2Pi(v1: float) -> float:
    """Wraps angles to [0, 2*pi]

    Parameters
    ----------
    v1 : np.ndarray
        Nx1 vector of angles [radians]

    Returns
    -------
    np.ndarray
        Nx1 vector of normalized angles [radians]
    """

    return np.mod(v1, two_pi)


# === WRAPTO2PI ===
@njit(cache=True, fastmath=True)
def wrapToPi(v1: float) -> float:
    """Wraps angles to [-pi, pi]

    Parameters
    ----------
    v1 : np.ndarray
        Nx1 vector of angles [radians]

    Returns
    -------
    np.ndarray
        Nx1 vector of normalized angles [radians]
    """
    v1 = wrapTo2Pi(v1)
    if isinstance(v1, float):
        if v1 > np.pi:
            v1 -= two_pi
    else:
        idx = np.nonzero(v1 > np.pi)[0]
        v1[idx] -= two_pi
    return v1


# === WRAPEULERANGLES ===
@njit(cache=True, fastmath=True)
def wrapEulerAngles(e: np.ndarray) -> np.ndarray:
    """Wraps angles to [-pi, pi]

    Parameters
    ----------
    e : np.ndarray
        3x1 vector of euler angles (roll-pitch-yaw) [radians]

    Returns
    -------
    np.ndarray
        3x1 vector of normalized euler angles (roll-pitch-yaw) [radians]
    """
    e1, e2, e3 = e
    if e2 > half_pi:
        e2 = np.pi - e2
        e1 = e1 + np.pi
        e3 = e3 + np.pi
    elif e2 < -half_pi:
        e2 = -np.pi - e2
        e1 = e1 + np.pi
        e3 = e3 + np.pi
    e1 = wrapToPi(e1)
    e3 = wrapToPi(e3)

    return np.array([e1, e2, e3])
