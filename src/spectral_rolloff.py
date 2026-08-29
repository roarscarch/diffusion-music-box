import numpy as np


def spectral_rolloff(spectrogram, sample_rate, percentile=0.85):
    """Compute the spectral rolloff frequency for each frame.

    The spectral rolloff is the frequency below which a given percentage
    (default 85%) of the total spectral energy is contained. It is a
    measure of the spectral shape and can be used to distinguish between
    bright and dark timbres.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (n_frames, n_freq_bins) or (n_freq_bins, n_frames).
        The spectrogram magnitude values. If shape is (n_freq_bins, n_frames),
        it will be transposed internally.
    sample_rate : int
        Sample rate of the audio signal in Hz.
    percentile : float, optional
        The percentile (0.0 to 1.0) of energy to use as the rolloff threshold.
        Default is 0.85.

    Returns
    -------
    np.ndarray
        1D array of length n_frames with the rolloff frequency in Hz for each frame.

    Raises
    ------
    ValueError
        If spectrogram is not 2D or percentile is not in (0, 1).
    """
    spectrogram = np.asarray(spectrogram, dtype=np.float32)
    if spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be 2D")
    if not 0.0 < percentile < 1.0:
        raise ValueError("Percentile must be between 0 and 1")

    # Assume frames are rows; if not, transpose
    if spectrogram.shape[0] > spectrogram.shape[1]:
        spectrogram = spectrogram.T

    n_frames, n_freq_bins = spectrogram.shape

    # Compute cumulative energy along frequency axis
    energy = np.sum(spectrogram, axis=1)
    cumulative = np.cumsum(spectrogram, axis=1)

    # Find the first bin where cumulative energy exceeds the threshold
    threshold = energy[:, np.newaxis] * percentile
    rolloff_bin = np.argmax(cumulative >= threshold, axis=1)

    # If no bin exceeds threshold (e.g., all zero), set to max bin
    # argmax returns 0 for all-zero rows, so handle that case
    all_zero = energy == 0
    rolloff_bin[all_zero] = n_freq_bins - 1

    # Convert bin index to frequency
    freq_per_bin = sample_rate / (2 * (n_freq_bins - 1)) if n_freq_bins > 1 else 0
    rolloff_freq = rolloff_bin * freq_per_bin

    return rolloff_freq
