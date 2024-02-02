import numpy as np
from numba import njit
from typing import Callable, Generator


# TODO: figure out where to sort functions

def parcorr(
    sequence: np.array,
    baseline_sequence: np.array,
    fft: Callable[[np.array], np.array] = np.fft.fft,
    ifft: Callable[[np.array], np.array] = np.fft.ifft,
) -> np.array:
    """performs correlation in parallel

    Parameters
    ----------
    sequence : np.array
        sequence to correlate with
    baseline_sequence : np.array
        sequence to correlate against
    fft : Callable[[np.array], np.array], optional
        fft function or object, by default np.fft.fft
    ifft : Callable[[np.array], np.array], optional
        ifft function or object, by default np.fft.ifft

    Returns
    -------
    np.array
        correlation power
    """
    correlation_fft = fft(sequence) * np.conj(fft(baseline_sequence))
    correlation_ifft = ifft(correlation_fft)
    correlation = np.abs(correlation_ifft) ** 2  # correlation power

    return correlation


def pcps(
    baseband_signals: np.array,
    code_replicas: np.array,
    baseband_fft: Callable[[np.array], np.array] = np.fft.fft,
    code_fft: Callable[[np.array], np.array] = np.fft.fft,
    ifft: Callable[[np.array], np.array] = np.fft.ifft,
) -> Generator[np.array, None, None]:
    """parallel code phase search across code replicas

    Parameters
    ----------
    baseband_signals : np.array
        baseband signal to correlate with
    code_replicas : np.array
        upsampled code replicas to correlate against
    baseband_fft : Callable[[np.array], np.array], optional
        fft function or object, by default np.fft.fft
    code_fft : Callable[[np.array], np.array], optional
       fft function or object, by default np.fft.fft
    ifft : Callable[[np.array], np.array], optional
        ifft function or object, by default np.fft.ifft

    Returns
    -------
    np.array
        correlation planes

    Yields
    ------
    Generator[np.array, None, None]
        correlation plane for each code replica and baseband signal
    """
    code_replica_cffts = (
        np.conj(code_fft(code_replica)) for code_replica in code_replicas
    )
    baseband_signals_fft = baseband_fft(baseband_signals)

    correlations = (
        pcps_parcorr(
            baseband_signals_fft=baseband_signals_fft,
            code_replica_cfft=code_replica_cfft,
        )
        for code_replica_cfft in code_replica_cffts
    )

    def pcps_parcorr(
        baseband_signals_fft: np.array, code_replica_cfft: np.array
    ) -> np.array:
        correlation_fft = baseband_signals_fft * code_replica_cfft
        correlation_ifft = ifft(correlation_fft)
        correlation = np.abs(correlation_ifft) ** 2

        return correlation

    return correlations


@njit(cache=True)
def upsample_sequence(
    sequence: np.array,
    nsamples: int,
    fsamp: float,
    fchip: float,
    phase_shift: float = 0.0,
    start_phase: float = 0.0,
) -> np.array:
    """upsample and phase shift sequence

    Parameters
    ----------
    sequence : np.array
        sequence to upsample
    nsamples : int
        # to upsample to
    fsamp : float
        sampling frequency [Hz]
    fchip : float
        chipping frequency of sequence
    phase_shift : float, optional
        phase basis for sequence if continually upsampling, by default 0.0
    start_phase : float, optional
        fractional starting phase that is calculated when continually upsampling, by default 0.0

    Returns
    -------
    np.array
        upsampled and phase shifted sequence
    """
    start_phase = phase_shift + start_phase
    phases = np.arange(0, nsamples) * (fchip / fsamp) + start_phase  # [chips]
    samples = sequence[phases.astype(np.int32) % sequence.size]

    return samples


@njit(cache=True)
def carrier_replica(
    fcarrier: float, nsamples: int, fsamp: float, start_phase: float = 0.0
) -> np.array:
    """generates carrier replica used to wipe off raw signal samples

    Parameters
    ----------
    fcarrier : float
        carrier frequency [Hz]
    nsamples : int
        # to upsample to
    fsamp : float
        sampling frequency [Hz]
    start_phase : float, optional
        fractional starting phase that is calculated when continually upsampling, by default 0.0

    Returns
    -------
    np.array
        carrier replica
    """
    phases = np.arange(0, nsamples) * fcarrier * (1 / fsamp) + start_phase  # [cycles]
    replica = np.exp(2 * np.pi * -1j * phases)

    return replica


@njit(cache=True)
def carrier_from_phase(
    delta_phase: float, nsamples: int, start_phase: float = 0.0
) -> np.array:
    """same functionality as carrier_replica with different interface and a non-conjugated output
    (useful for accurately simulating phase instead of integrating frequency)

    Parameters
    ----------
    delta_phase : float
        total change in phase between updates
    nsamples : int
        # of samples in block
    start_phase : float, optional
        fractional starting phase that is calculated when continually upsampling, by default 0.0

    Returns
    -------
    np.array
        carrier wave
    """
    phases = np.arange(0, nsamples) * (delta_phase / nsamples) + start_phase  # [cycles]
    replica = np.exp(2 * np.pi * 1j * phases)

    return replica


@njit(cache=True)
def carrier_from_frequency(
    fcarrier: float,
    fsamp: float,
    duration: float,
    fcarrier_rate: float = 0.0,
) -> np.array:
    """generates carrier signal for modulation

    Parameters
    ----------
    fcarrier : float
        carrier frequency [Hz]
    fsamp : float
        sampling frequency [Hz]
    duration : float
        duration of signal [s]
    fcarrier_rate : float, optional
        Doppler rate of signal [Hz/s], by default 0.0

    Returns
    -------
    np.array
        carrier signal
    """
    time = np.arange(0, fsamp * duration) * (1 / fsamp)
    phases = fcarrier * time + 0.5 * fcarrier_rate * time**2
    carrier = np.exp(2 * np.pi * 1j * phases)

    return carrier


@njit(cache=True)
def apply_complex_cn0(samples: np.array, cn0: float, fsamp: float) -> np.array:
    """applies corresponding noise and amplitude to signal samples based on C/N0

    Parameters
    ----------
    samples : np.array
        samples to add noise to
    cn0 : float
        carrier-to-noise density ratio [dB-Hz]
    fsamp : float
        sampling frequency [Hz]

    Returns
    -------
    np.array
        noisy samples
    """
    cn0 = 10 ** (cn0 / 10)  # linear ratio

    A = np.sqrt((2 * cn0) / fsamp)
    noise = np.random.randn(2 * samples.size).view(samples.dtype) / np.sqrt(2)

    noisy_samples = A * samples + noise

    return noisy_samples


@njit(cache=True)
def apply_real_cn0(samples: np.array, cn0: float, fsamp: float) -> np.array:
    """applies corresponding noise and amplitude to signal samples based on C/N0

    Parameters
    ----------
    samples : np.array
        samples to add noise to
    cn0 : float
        carrier-to-noise density ratio [dB-Hz]
    fsamp : float
        sampling frequency [Hz]

    Returns
    -------
    np.array
        noisy samples
    """
    cn0 = 10 ** (cn0 / 10)  # linear ratio

    A = 2 * np.sqrt(cn0 / fsamp)
    noise = np.random.randn(samples.size).astype(samples.dtype)

    noisy_samples = A * samples + noise

    return noisy_samples


@njit(cache=True)
def quantize(samples: np.array, bit_depth: int, headroom: int = 3):
    scale_factor = 0.5 * ((2**bit_depth) - 1 - headroom) / np.max(np.abs(samples))
    quantized_samples = scale_factor * samples

    return quantized_samples
