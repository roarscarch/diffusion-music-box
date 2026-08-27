import numpy as np


def spectral_rolloff(spectrogram, sample_rate, threshold=0.85):
    """Compute the spectral rolloff point for each frame.

    The spectral rolloff is the frequency below which a given percentage
    (threshold) of the total spectral energy is concentrated. It is a useful
    feature for characterizing the brightness or timbre of a signal.

    Parameters
    ----------
    spectrogram : np.ndarray
        Magnitude spectrogram of shape (n_frames, n_freq_bins) or
        (n_freq_bins, n_frames). The function detects orientation based on
        the typical FFT layout (freq bins as rows) but handles both.
    sample_rate : int
        Sample rate of the audio signal in Hz.
    threshold : float, optional
        Fraction of total energy (0 < threshold < 1) below which the rolloff
        point is defined. Default is 0.85 (common value).

    Returns
    -------
    np.ndarray
        Array of rolloff frequencies in Hz for each frame, shape (n_frames,).
    """
    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be between 0 and 1 exclusive")

    # Ensure spectrogram is 2D
    if spectrogram.ndim != 2:
        raise ValueError("spectrogram must be 2D")

    # Detect orientation: assume frequency bins are the larger dimension
    # and time frames are the smaller. This is a heuristic; for shape
    # (n_freq, n_time) we transpose internally.
    if spectrogram.shape[0] < spectrogram.shape[1]:
        # Likely (n_frames, n_freq) -> transpose to (n_freq, n_frames)
        spec = spectrogram.T
    else:
        spec = spectrogram

    n_freq_bins, n_frames = spec.shape

    # Frequency resolution
    fft_size = (n_freq_bins - 1) * 2
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    if len(freqs) != n_freq_bins:
        # Fallback: linearly spaced bins (approximation)
        freqs = np.linspace(0, sample_rate / 2, n_freq_bins)

    # Compute cumulative energy
    energy = np.sum(spec ** 2, axis=0)
    total_energy = np.sum(energy)
    if total_energy == 0:
        return np.zeros(n_frames)

    cumsum = np.cumsum(energy)
    rolloff_idx = np.searchsorted(cumsum, threshold * total_energy)

    # Rolloff frequency at that index
    rolloff_freq = freqs[rolloff_idx]

    return rolloff_freq
