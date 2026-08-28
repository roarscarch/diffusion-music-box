import numpy as np


def spectral_centroid(spectrogram, sample_rate=22050, fft_size=1024):
    """Compute the spectral centroid for each time frame of a spectrogram.

    The spectral centroid is a measure of the 'brightness' of the spectrum,
    defined as the weighted mean of frequency bins, where weights are the
    magnitudes. It is useful for analyzing timbral characteristics of audio.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (freq_bins, time_frames) containing magnitudes.
    sample_rate : int, optional
        Sample rate of the audio (used to compute bin frequencies).
    fft_size : int, optional
        FFT size that produced the spectrogram (for bin spacing).

    Returns
    -------
    np.ndarray
        1D array of shape (time_frames,) with centroid values in Hz.
    """
    if spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be 2D")
    freq_bins = spectrogram.shape[0]
    # Compute bin frequencies in Hz
    freqs = np.linspace(0, sample_rate / 2, freq_bins)
    # Weighted average of frequencies by magnitude
    total = np.sum(spectrogram, axis=0)
    # Avoid division by zero
    total = np.where(total == 0, 1e-12, total)
    centroid = np.sum(spectrogram * freqs[:, np.newaxis], axis=0) / total
    return centroid
