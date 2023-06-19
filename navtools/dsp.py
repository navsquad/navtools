import numpy as np
import numpy.matlib
from numba import njit, prange


def parcorr(baseband_signal: np.array, replica: np.array) -> np.array:
    """Correlate baseband signal and replica in parallel using FFT.

    Parameters
    ----------
    baseband_signal : np.array
        Baseband sequence (eg. received GPS signal)
    replica : np.array
        Replica of signal within baseband signal (eg. GPS PRN)

    Returns
    -------
    np.array
        Correlation values across sample lags
    """
    replica_pad_size = baseband_signal.size - replica.size
    padded_replica = np.pad(replica, (0, replica_pad_size))

    baseband_fft = np.fft.fft(baseband_signal)
    replica_fft = np.fft.fft(replica)

    correlation_fft = np.multiply(baseband_fft, np.conjugate(replica_fft))
    correlation_ifft = np.fft.ifft(correlation_fft)
    correlation = np.power(np.abs(correlation_ifft), 2)

    return correlation


def vparcorr(baseband_signal: np.array, replica: np.array) -> np.array:
    """Correlate baseband signal and replica in parallel using FFT.

    Parameters
    ----------
    baseband_signal : np.array
        Baseband sequence (eg. received GPS signal)
    replica : np.array
        Replica of signal within baseband signal (eg. GPS PRN)

    Returns
    -------
    np.array
        Correlation values across sample lags
    """

    baseband_fft = np.fft.fft(baseband_signal, axis=1)
    replica_fft = np.fft.fft(replica)

    correlation_fft = baseband_fft * np.conjugate(replica_fft)
    correlation_ifft = np.fft.ifft(correlation_fft, axis=1)
    correlation = np.power(np.abs(correlation_ifft), 2)

    return correlation


def fft(signal, fsamp):
    num_samples = signal.size
    fft = np.fft.fftshift(np.fft.fft(signal))
    frequency_range = np.arange(-fsamp / 2, fsamp / 2, fsamp / num_samples)
    return fft, frequency_range


def carrier_replica(fcarr, fsamp, num_samples, rem_phase=0):
    psamp = 1 / fsamp
    sample_range = np.arange(0, num_samples)
    time = psamp * sample_range
    phase = fcarr * time + rem_phase

    replica = np.exp(2 * np.pi * phase)
    return replica


def vcarrier_replica(fcarr, fsamp, num_samples, rem_phase=0):
    psamp = 1 / fsamp
    sample_range = np.arange(0, num_samples)
    time = psamp * sample_range
    phase = np.expand_dims(fcarr, axis=1) * np.expand_dims(time, axis=0) + rem_phase

    replica = np.exp(2 * np.pi * phase)
    return replica


def upsample_sequence(
    sequence: np.array, fsamp, fchip, upsample_size=None, rem_phase=0, chip_shift=0
):
    samples_per_chip = fsamp / fchip
    chips_per_sample = 1 / samples_per_chip
    chip_phase = rem_phase + chip_shift
    extended_code = np.concatenate([[sequence[-1]], sequence, [sequence[0]]])

    if upsample_size is None:
        upsample_size = np.ceil((sequence.size - chip_phase) * samples_per_chip).astype(
            int
        )  # samples per code period

    fractional_chip_index = np.arange(0, upsample_size) * chips_per_sample + chip_phase
    whole_chip_index = np.mod(np.ceil(fractional_chip_index), sequence.size).astype(int)

    upsampled_code = extended_code[
        np.where(whole_chip_index == 0, sequence.size, whole_chip_index)
    ]
    rem_phase = fractional_chip_index[-1] + chips_per_sample - sequence.size

    return upsampled_code, rem_phase


def pcps(signal_samples, prn_replica, frange, fsamp):
    results = np.empty([frange.size, signal_samples.size])

    for index, freq in enumerate(frange):
        psamp = 1 / fsamp
        sample_range = np.arange(0, signal_samples.size)
        phase = freq * psamp * sample_range
        replica = np.exp(2 * np.pi * -1j * phase)

        baseband_signal = signal_samples * replica
        correlation = parcorr(baseband_signal, prn_replica)
        results[index, :] = correlation

    return results


def pcps3(signal_samples, code_replica, frange, fsamp):
    replica = vcarrier_replica(
        fsamp=fsamp, num_samples=signal_samples.size, fcarr=frange
    )
    baseband_signal = signal_samples * replica
    correlation = vparcorr(baseband_signal=baseband_signal, replica=code_replica)

    return correlation
