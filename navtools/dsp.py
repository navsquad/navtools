import numpy as np


def parcorr(
    sequence: np.array, baseline_sequence: np.array, fft=np.fft.fft, ifft=np.fft.ifft
) -> np.array:
    """parallel correlation of a sequence with a baseline sequence using fft and ifft of user's choice

    Parameters
    ----------
    sequence : np.array
        sequence to correlate against (eg. raw signal data)
    baseline_sequence : np.array
        sequence to correlate with (eg. prn replica)
    fft : _type_, optional
        fft function, by default np.fft.fft
    ifft : _type_, optional
        ifft function, by default np.fft.ifft

    Returns
    -------
    np.array
        correlation magnitudes for each index of baseline sequence
    """

    correlation_fft = fft(sequence) * np.conj(fft(baseline_sequence))
    correlation_ifft = ifft(correlation_fft)
    correlation = np.abs(correlation_ifft) ** 2

    return correlation


def pcps(
    code_replicas: np.array,
    baseband_signals: np.array,
    code_fft=np.fft.fft,
    baseband_fft=np.fft.fft,
    ifft=np.fft.ifft,
) -> np.array:
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

    def pcps_parcorr(baseband_signals_fft: np.array, code_replica_cfft: np.array):
        """parallel correlation (see :func:`navtools.dsp.parcorr`) where parameters are outputs of different FFTs"""

        correlation_fft = baseband_signals_fft * code_replica_cfft
        correlation_ifft = ifft(correlation_fft)
        correlation = np.abs(correlation_ifft) ** 2

        return correlation

    return correlations


def upsample_sequence(
    sequence: np.array, fsamp, fchip, upsample_size=None, rem_phase=0, chip_shift=0
):
    # TODO: add comments to make clear
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
