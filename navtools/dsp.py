import numpy as np


def parcorr(
    baseband_signal: np.array, conj_replica_fft: np.array, fft_object, ifft_object
) -> np.array:
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

    baseband_fft = fft_object(baseband_signal)

    correlation_fft = baseband_fft * conj_replica_fft
    correlation_ifft = ifft_object(correlation_fft)
    correlation = np.abs(correlation_ifft) ** 2

    return correlation


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


def pcps(signal_samples, sample_range, conj_prn_replica_fft, fft, ifft, freq, fsamp):
    psamp = 1 / fsamp
    phase = freq * psamp * sample_range
    replica = np.exp(2 * np.pi * -1j * phase, dtype=signal_samples.dtype)

    baseband_signal = signal_samples * replica
    correlation = parcorr(baseband_signal, conj_prn_replica_fft, fft, ifft)

    return correlation
