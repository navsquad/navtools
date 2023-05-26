import numpy as np


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
    replica_fft = np.fft.fft(padded_replica)

    correlation_fft = np.multiply(baseband_fft, np.conjugate(replica_fft))
    correlation_ifft = np.fft.ifft(correlation_fft)
    correlation = np.power(np.abs(correlation_ifft), 2)

    return correlation
