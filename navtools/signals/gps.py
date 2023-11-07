import numpy as np

from navtools.signals.types import PhaseShiftKeyedSignal


# Taps
L1CA_TAPS = {
    1: [2, 6],
    2: [3, 7],
    3: [4, 8],
    4: [5, 9],
    5: [1, 9],
    6: [2, 10],
    7: [1, 8],
    8: [2, 9],
    9: [3, 10],
    10: [2, 3],
    11: [3, 4],
    12: [5, 6],
    13: [6, 7],
    14: [7, 8],
    15: [8, 9],
    16: [9, 10],
    17: [1, 4],
    18: [2, 5],
    19: [3, 6],
    20: [4, 7],
    21: [5, 8],
    22: [6, 9],
    23: [1, 3],
    24: [4, 6],
    25: [5, 7],
    26: [6, 8],
    27: [7, 9],
    28: [8, 10],
    29: [1, 6],
    30: [2, 7],
    31: [3, 8],
    32: [4, 9],
}



def gps_l1ca_prn_generator(prn: int):
    G1 = 0xFFFF
    G2 = 0xFFFF
    
    
    
    


# PRN Generators
def gps_l1ca_prn_generator(prn: int):
    # TODO: rewrite with bit-wise operators
    lsfr_taps = [
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 8],
        [0, 8],
        [1, 9],
        [0, 7],
        [1, 8],
        [2, 9],
        [1, 2],
        [2, 3],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [0, 2],
        [3, 5],
        [4, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [0, 5],
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
        [3, 9],
        [0, 6],
        [1, 7],
        [3, 9],
    ]

    tap = np.array(lsfr_taps)

    # G1 LFSR: x^10 + x^3 + 1
    s = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    n = s.size
    g1 = np.ones(n)
    L = 2**n - 1

    # G2j LFSR: x^10 + x^9 + x^8 + x^6 + x^2 + 1
    t = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    q = np.ones(n)

    # generate C/A code sequence
    tap_sel = tap[prn - 1, :]
    g2 = np.zeros(L)
    g = np.zeros(L)

    for inc in range(L):
        g2[inc] = np.mod(np.sum(q[tap_sel]), 2)
        g[inc] = np.mod(g1[n - 1] + g2[inc], 2)
        g1 = np.delete(np.insert(g1, 0, np.mod(np.sum(g1 * s), 2)), -1)
        q = np.delete(np.insert(q, 0, np.mod(np.sum(q * t), 2)), -1)

    return g


# Signals
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
