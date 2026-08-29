import numpy as np


def spectral_centroid(spectrogram, sample_rate=None, fft_size=None, hop_length=None):
    """Compute the spectral centroid for each frame of a spectrogram.

    The spectral centroid is the weighted mean of the frequencies present in
    the signal, where the weights are the magnitudes of the corresponding
    frequency bins. It is a measure of the 'brightness' of the sound, and is
    commonly used in audio analysis and music information retrieval.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (freq_bins, n_frames) containing magnitude or power
        values. It can be the output of a short-time Fourier transform (STFT).
    sample_rate : int, optional
        Sample rate of the audio signal in Hz. If provided, the centroid is
        returned in Hertz; otherwise, it is returned in FFT bins.
    fft_size : int, optional
        FFT size used to generate the spectrogram. Required if sample_rate is
        given and the frequency resolution is needed. If None, it is inferred
        from the spectrogram shape: fft_size = 2 * (freq_bins - 1).
    hop_length : int, optional
        Hop length in samples. Not used directly, but kept for API consistency.

    Returns
    -------
    np.ndarray
        1D array of length n_frames with the spectral centroid values.
        If sample_rate is provided, values are in Hz; otherwise in bins.
    """
    if spectrogram.ndim != 2:
        raise ValueError("spectrogram must be a 2D array")

    n_freq_bins, n_frames = spectrogram.shape
    if n_freq_bins < 2:
        raise ValueError("spectrogram must have at least 2 frequency bins")

    # Magnitude spectrogram (ensure non-negative)
    magnitudes = np.abs(spectrogram).astype(np.float64)

    # Frequency bin indices
    bin_indices = np.arange(n_freq_bins, dtype=np.float64)

    # Compute weighted mean per frame
    total_magnitude = np.sum(magnitudes, axis=0)
    # Avoid division by zero; set centroid to 0 for silent frames
    total_magnitude[total_magnitude == 0] = 1.0
    centroid_bins = np.sum(bin_indices[:, np.newaxis] * magnitudes, axis=0) / total_magnitude

    if sample_rate is not None:
        # Convert bin index to frequency
        if fft_size is None:
            fft_size = 2 * (n_freq_bins - 1)
        frequency_resolution = sample_rate / fft_size
        centroid_hz = centroid_bins * frequency_resolution
        return centroid_hz.astype(np.float32)
    else:
        return centroid_bins.astype(np.float32)
