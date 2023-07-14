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


def upsample_sequence(
    sequence: np.array, fsamp, fchip, upsample_size=None, rem_phase=0, chip_shift=0
):
    samples_per_chip = fsamp / fchip
    chips_per_sample = 1 / samples_per_chip
    chip_phase = rem_phase + chip_shift
    samples_per_code = np.ceil((sequence.size - chip_phase) * samples_per_chip).astype(
        int
    )

    if upsample_size is None:
        upsample_size = samples_per_code  # samples per code period
    multiple = upsample_size / samples_per_code
    sequence = repeat_sequence(sequence=sequence, multiple=multiple)

    extended_code = np.concatenate([[sequence[-1]], sequence, [sequence[0]]])

    fractional_chip_index = np.arange(0, upsample_size) * chips_per_sample + chip_phase
    whole_chip_index = np.mod(np.ceil(fractional_chip_index), sequence.size).astype(int)

    upsampled_code = extended_code[
        np.where(whole_chip_index == 0, sequence.size, whole_chip_index)
    ]
    rem_phase = fractional_chip_index[-1] + chips_per_sample - sequence.size

    return upsampled_code, rem_phase


def repeat_sequence(sequence: np.array, multiple: float):
    integer_sequence_multiples = np.tile(sequence, np.fix(multiple).astype(int))
    fractional_index = np.round(sequence.size * np.mod(multiple, 1)).astype(int)
    fractional_sequence = sequence[:fractional_index]

    return np.concatenate([integer_sequence_multiples, fractional_sequence])


def pcps(
    prn_replica_cfft,
    baseband_signals_a,
    baseband_signals_b,
    fft,
    ifft,
):
    correlation_a = np.array(
        [
            parcorr(
                baseband_signal=baseband_signal,
                conj_replica_fft=prn_replica_cfft,
                fft_object=fft,
                ifft_object=ifft,
            )
            for baseband_signal in baseband_signals_a
        ]
    )
    correlation_b = np.array(
        [
            parcorr(
                baseband_signal=baseband_signal,
                conj_replica_fft=prn_replica_cfft,
                fft_object=fft,
                ifft_object=ifft,
            )
            for baseband_signal in baseband_signals_b
        ]
    )

    return correlation_a, correlation_b
