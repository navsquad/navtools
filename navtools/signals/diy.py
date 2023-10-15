""" diy.py contains do-it-yourself, mock signal properties (eg. BPSK "freedom" signal)"""

import numpy as np
from scipy.signal import max_len_seq

from navtools.signals.types import PhaseShiftKeyedSignal


### prn generators ###
def freedom_prn_generator(prn):
    taps_a = [7, 6, 4, 2, 1]
    taps_b = [8, 5, 1]

    state = np.ones(10)

    sequence_a = max_len_seq(nbits=10, taps=taps_a, state=state)[0]
    sequence_b = max_len_seq(nbits=10, taps=taps_b, state=state)[0]

    code = 2 * np.logical_xor(sequence_a, np.roll(sequence_b, prn)) - 1

    return code


### signals ###
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
