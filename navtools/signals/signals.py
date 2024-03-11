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
    
    
@dataclass(frozen=True)
class SignalOfOpportunity(SatelliteSignal):
    fcarrier: float
    wavelength: float


@njit(cache=True)
def bpsk_correlator(
    T: float,
    cn0: float,
    chip_error: float,
    ferror: float,
    phase_error: float,
    tap_spacing: float = 0,
    include_noise: bool = True,
):
    cn0 = nt.atleast_1d(cn0)
    chip_error = nt.atleast_1d(chip_error)
    ferror = nt.atleast_1d(ferror)
    phase_error = nt.atleast_1d(phase_error)

    # handle dimensions for broadcasting
    chip_error = nt.smart_transpose(col_size=cn0.size, transformed_array=chip_error)
    ferror = nt.smart_transpose(col_size=cn0.size, transformed_array=ferror)
    phase_error = nt.smart_transpose(col_size=cn0.size, transformed_array=phase_error)

    cn0 = 10 ** (cn0 / 10)  # linear ratio
    amplitude = np.sqrt(2 * cn0 * T) * np.sinc(np.pi * ferror * T)

    acorr_magnitude = 1 - np.abs(chip_error - tap_spacing)
    acorr_magnitude = np.where(acorr_magnitude < 0, 0.0, acorr_magnitude)

    correlator = (
        amplitude
        * acorr_magnitude
        * np.exp(np.pi * 1j * (ferror * T + 2 * phase_error))
    )

    inphase = np.real(correlator)
    quadrature = np.imag(correlator)

    if include_noise:
        inphase += np.random.randn(*correlator.shape)
        quadrature += np.random.randn(*correlator.shape)

    return inphase, quadrature
