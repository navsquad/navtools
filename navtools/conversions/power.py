"""
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
"""

__all__ = ["cn02snr", "snr2cn0"]

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def cn02snr(cn0: float, front_end_bw: float = 4e6, noise_figure: float = 0.0) -> float:
    """Convert Carrier-to-Noise Ratio into raw Signal-to-Noise Ratio

    Parameters
    ----------
    cn0 : float
        Carrier-to-Noise ratio [dB/Hz]
    front_end_bw : float, optional
        Front end receiver bandwidth [Hz], by default 4e6
    noise_figure : float, optional
        Noise losses [W], by default 0.0

    Returns
    -------
    float
        Signal-to-Noise ratio [dB]
    """
    snr = cn0 - 10 * np.log10(front_end_bw) - noise_figure  # dB
    return snr


@njit(cache=True, fastmath=True)
def snr2cn0(snr: float, front_end_bw: float = 4e6, noise_figure: float = 0.0) -> float:
    """Convert raw Signal-to-Noise Ratio into Carrier-to-Noise Ratio

    Parameters
    ----------
    snr : float
        Signal-to-Noise ratio [dB]
    front_end_bw : float, optional
        Front end receiver bandwidth [Hz], by default 4e6
    noise_figure : float, optional
        Noise losses [W], by default 0.0

    Returns
    -------
    float
        Carrier-to-Noise ratio [dB/Hz]
    """
    cn0 = snr + 10 * np.log10(front_end_bw) + noise_figure  # dB-Hz
    return cn0


# @njit(cache=True, fastmath=True)
# def dB2power(dB: float) -> float:
#   """Convert dB to its power not in a log scale

#   Parameters
#   ----------
#   dB : float
#       power in log scale dB

#   Returns
#   -------
#   float
#       raw power
#   """
#   return 10**(0.1*dB)


# @njit(cache=True, fastmath=True)
# def power2dB(p: float, T: float) -> float:
#   """Convert power to a log scale

#   Parameters
#   ----------
#   p : float
#       raw power
#   T : float
#       integration time [T]

#   Returns
#   -------
#   float
#       power in log scale dB
#   """
#   return 10*np.log10(p/T)
