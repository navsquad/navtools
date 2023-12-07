import numpy as np
from dataclasses import dataclass
from numba import njit


@dataclass(frozen=True)
class SatelliteSignal:
    transmit_power: float  # [W]
    transmit_antenna_gain: float  # [dBi]


@dataclass(frozen=True)
class PhaseShiftKeyedSignal(SatelliteSignal):
    fcarrier: float

    fbit_data: float
    msg_length_data: int
    fchip_data: float
    code_length_data: int
    prn_generator_data: any

    fbit_pilot: float = None
    msg_length_pilot: int = None
    fchip_pilot: float = None
    code_length_pilot: float = None
    prn_generator_pilot: any = None


@njit(cache=True)
def bpsk_correlator(
    T: float,
    chip_error: float,
    ferror: float,
    phase_error: float,
    chip_offset: float = 0,
):
    correlator = (1 - np.abs(chip_error + chip_offset)) * np.exp(
        np.pi * 1j * (ferror * T + 2 * phase_error)
    )

    inphase = np.real(correlator)
    quadrature = np.imag(correlator)

    return inphase, quadrature
