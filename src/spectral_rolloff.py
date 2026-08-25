import numpy as np


def spectral_rolloff(spectrogram, sample_rate=22050, threshold=0.85):
    """Compute the spectral rolloff frequency for each time frame.

    The spectral rolloff is the frequency below which a specified percentage
    (default 85%) of the total spectral energy is contained. It provides a
    measure of the spectral shape and brightness of an audio signal.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (freq_bins, time_frames) containing magnitude
        spectrogram values (non-negative).
    sample_rate : int, optional
        Sample rate of the audio signal, used to convert bin indices to Hz.
    threshold : float, optional
        Fraction of total energy below which the rolloff frequency is found.
        Must be between 0 and 1.

    Returns
    -------
    np.ndarray
        1D array of length time_frames containing the rolloff frequency in Hz
        for each frame.

    Raises
    ------
    ValueError
        If threshold is not between 0 and 1.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    # Ensure non-negative values
    spec = np.abs(spectrogram)
    total_energy = np.sum(spec, axis=0)
    cumulative = np.cumsum(spec, axis=0)

    # Find the smallest bin where cumulative energy exceeds threshold * total
    # For frames with zero energy, rolloff is 0 Hz.
    freq_bins = spec.shape[0]
    rolloff_bin = np.zeros(spec.shape[1], dtype=int)

    for i in range(spec.shape[1]):
        if total_energy[i] == 0:
            rolloff_bin[i] = 0
        else:
            # Find first index where cumulative >= threshold * total
            rolloff_bin[i] = np.searchsorted(cumulative[:, i], threshold * total_energy[i])

    # Convert bin index to frequency in Hz
    # Frequency of bin k = k * sample_rate / (2 * (freq_bins - 1)) for one-sided spectrum
    # More commonly, bin width = sample_rate / (2 * (freq_bins - 1)) for a one-sided FFT.
    # For simplicity, assume uniform spacing from 0 to Nyquist.
    if freq_bins > 1:
        nyquist = sample_rate / 2.0
        freqs = np.linspace(0.0, nyquist, freq_bins)
    else:
        freqs = np.array([0.0])

    rolloff_freqs = freqs[rolloff_bin]
    return rolloff_freqs.astype(np.float32)
