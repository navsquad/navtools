import numpy as np
import navtools as nt
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


# @njit(cache=True)
def bpsk_correlator(
    T: float,
    cn0: float,
    chip_error: float,
    ferror: float,
    phase_error: float,
    tap_spacing: float = 0,
    include_noise: bool = True,
):
    # handle dimensions for broadcasting
    if isinstance(cn0, np.ndarray):
        chip_error = nt.smart_transpose(col_size=cn0.size, transformed_array=chip_error)
        ferror = nt.smart_transpose(col_size=cn0.size, transformed_array=ferror)
        phase_error = nt.smart_transpose(
            col_size=cn0.size, transformed_array=phase_error
        )

        size = cn0.size
    else:
        size = 1

    cn0 = 10 ** (cn0 / 10)  # linear ratio
    amplitude = np.sqrt(2 * cn0 * T) * np.sinc(np.pi * ferror * T)

    acorr_magnitude = 1 - np.abs(chip_error - tap_spacing)
    acorr_magnitude = np.where(acorr_magnitude < 0, 0.0, acorr_magnitude)

    correlator = (
        amplitude
        * acorr_magnitude
        * np.exp(np.pi * 1j * (ferror * T + 2 * phase_error))
    )

    if include_noise:
        inphase_noise = np.random.randn(size)
        quadrature_noise = np.random.randn(size)
    else:
        inphase_noise = 0.0
        quadrature_noise = 0.0

    inphase = np.real(correlator) + inphase_noise
    quadrature = np.imag(correlator) + quadrature_noise

    return inphase, quadrature
