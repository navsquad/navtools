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
    correlation = np.abs(correlation_ifft) ** 2  # correlation power

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
    sequence: np.array,
    nsamples: int,
    fsamp: float,
    fchip: float,
    phase_shift: float = 0.0,
    start_phase: float = 0.0,
):
    start_phase = phase_shift + start_phase
    phases = np.arange(0, nsamples) * (fchip / fsamp) + start_phase  # [chips]
    samples = sequence[phases.astype(int) % sequence.size]

    return samples


def carrier_replica(fcarrier: float, nsamples: int, fsamp: float, start_phase: float):
    phases = np.arange(0, nsamples) * fcarrier * (1 / fsamp) + start_phase  # [cycles]
    replica = np.exp(2 * np.pi * -1j * phases)

    return replica
