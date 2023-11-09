""" diy.py contains do-it-yourself, mock signal properties and prn generators (eg. BPSK FREEDOM signal)"""

import numpy as np
from numba import njit
from navtools.common import msequence

from navtools.signals.types import PhaseShiftKeyedSignal


# PRN Generators
@njit(cache=True)
def freedom_prn_generator(prn: int):
    A_TAPS = np.array([1, 2, 4, 6, 7, 10])
    B_TAPS = np.array([1, 5, 8, 10])

    sequence_a = msequence(nbits=10, taps=A_TAPS)
    sequence_b = msequence(nbits=10, taps=B_TAPS)

    code = 2 * np.logical_xor(sequence_a, np.roll(sequence_b, prn)) - 1

    return code


@njit(cache=True)
def auburn_prn_generator(prn: int):
    A_TAPS = np.array([1, 4, 6, 7, 9, 10])
    B_TAPS = np.array([1, 3, 4, 5, 6, 7, 8, 10])

    sequence_a = msequence(nbits=10, taps=A_TAPS)
    sequence_b = msequence(nbits=10, taps=B_TAPS)

    code = 2 * np.logical_xor(sequence_a, np.roll(sequence_b, prn)) - 1

    return code


# Signal Properties
FREEDOM = PhaseShiftKeyedSignal(
    transmit_power=25.5,  # minimum EIRP of Iridium-NEXT @ 90 degrees elevation is 38.5 dBW
    transmit_antenna_gain=13,
    fcarrier=1776.74e6,
    fbit_data=50.0,
    msg_length_data=1500,
    fchip_data=1.023e6,
    code_length_data=1023,
    prn_generator_data=freedom_prn_generator,
)

AUBURN = PhaseShiftKeyedSignal(
    transmit_power=25.5,  # minimum EIRP of Iridium-NEXT @ 90 degrees elevation is 38.5 dBW
    transmit_antenna_gain=13,
    fcarrier=1856.21e6,
    fbit_data=50.0,
    msg_length_data=1500,
    fchip_data=1.023e6,
    code_length_data=1023,
    prn_generator_data=auburn_prn_generator,
)
