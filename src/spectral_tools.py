import numpy as np


def to_log_spectrogram(spectrogram, eps=1e-10):
    """Convert a linear spectrogram to a log-magnitude spectrogram.

    Parameters
    ----------
    spectrogram : np.ndarray
        Linear magnitude spectrogram, shape (freq_bins, time_frames).
    eps : float, optional
        Small constant to avoid log(0).

    Returns
    -------
    np.ndarray
        Log-magnitude spectrogram with same shape.
    """
    return np.log10(np.abs(spectrogram) + eps)


def from_log_spectrogram(log_spec, eps=1e-10):
    """Convert a log-magnitude spectrogram back to linear magnitude.

    Parameters
    ----------
    log_spec : np.ndarray
        Log-magnitude spectrogram.
    eps : float, optional
        Small constant to avoid underflow.

    Returns
    -------
    np.ndarray
        Linear magnitude spectrogram.
    """
    return np.power(10.0, log_spec) - eps


def mel_filterbank(n_fft, sample_rate, n_mels=128, fmin=0.0, fmax=None):
    """Create a mel filterbank matrix.

    Parameters
    ----------
    n_fft : int
        FFT size.
    sample_rate : int
        Sample rate of the audio.
    n_mels : int, optional
        Number of mel bands.
    fmin : float, optional
        Minimum frequency.
    fmax : float, optional
        Maximum frequency. Defaults to Nyquist.

    Returns
    -------
    np.ndarray
        Filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    # Convert frequencies to mel scale
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Map frequencies to FFT bins
    bin_freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left = bin_indices[i]
        center = bin_indices[i + 1]
        right = bin_indices[i + 2]
        if left < center:
            filterbank[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if center < right:
            filterbank[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return filterbank


def apply_mel(spectrogram, filterbank):
    """Apply a mel filterbank to a spectrogram.

    Parameters
    ----------
    spectrogram : np.ndarray
        Linear magnitude spectrogram, shape (freq_bins, time_frames).
    filterbank : np.ndarray
        Mel filterbank matrix, shape (n_mels, freq_bins).

    Returns
    -------
    np.ndarray
        Mel-spectrogram of shape (n_mels, time_frames).
    """
    return filterbank @ spectrogram
