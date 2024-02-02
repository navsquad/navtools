import numpy as np
from numba import njit

from navtools.signals.gen.signals import PhaseShiftKeyedSignal
from navtools.common import get_bit_value, xor_register_taps


# Signal Taps
L1CA_TAPS = np.array(
    [
        [2, 6],
        [3, 7],
        [4, 8],
        [5, 9],
        [1, 9],
        [2, 10],
        [1, 8],
        [2, 9],
        [3, 10],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [6, 9],
        [1, 3],
        [4, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
    ]
)
L1CA_REGISTER_SIZE = 10
L1CA_G1_TAPS = np.array([3, 10])
L1CA_G2_TAPS = np.array([2, 3, 6, 8, 9, 10])


# PRN Generators
@njit(cache=True)
def gps_l1ca_prn_generator(prn: int):
    prn -= 1  # for zero-indexing
    prn_taps = L1CA_TAPS[prn]

    # initialize registers as ones
    G1 = 2**L1CA_REGISTER_SIZE - 1
    G2 = 2**L1CA_REGISTER_SIZE - 1

    code = []
    for _ in range(1023):
        # compute next chip in sequence
        chip = (
            get_bit_value(number=G1, index=0)  # [10]
            ^ get_bit_value(number=G2, index=L1CA_REGISTER_SIZE - prn_taps[0])
            ^ get_bit_value(number=G2, index=L1CA_REGISTER_SIZE - prn_taps[1])
        )
        code.append(chip)

        # calculate next bit in registers
        feedback1 = xor_register_taps(
            register=G1, nbits=L1CA_REGISTER_SIZE, taps=L1CA_G1_TAPS
        )
        feedback2 = xor_register_taps(
            register=G2, nbits=L1CA_REGISTER_SIZE, taps=L1CA_G2_TAPS
        )

        # shift and update registers
        G1 = (G1 >> 1) | (feedback1 << (L1CA_REGISTER_SIZE - 1))
        G2 = (G2 >> 1) | (feedback2 << (L1CA_REGISTER_SIZE - 1))

    code = 2 * np.array(code) - 1  # non-return to zero

    return code


# Signal Properties
L1CA = PhaseShiftKeyedSignal(
    transmit_power=16.5,
    transmit_antenna_gain=13,
    fcarrier=1575.42e6,
    fbit_data=50.0,
    msg_length_data=1500,
    fchip_data=1.023e6,
    code_length_data=1023,
    prn_generator_data=gps_l1ca_prn_generator,
)
