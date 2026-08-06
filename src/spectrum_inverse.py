import numpy as np


def normalize_spectrogram(tile, eps=1e-8):
    """Normalize a spectrogram tile to have zero mean and unit variance.

    Parameters
    ----------
    tile : np.ndarray
        Input spectrogram tile.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    (np.ndarray, tuple)
        Normalized tile and a tuple (mean, std) for inverse normalization.
    """
    mean = np.mean(tile)
    std = np.std(tile)
    if std < eps:
        std = 1.0
    normalized = (tile - mean) / (std + eps)
    return normalized, (mean, std)


def denormalize_spectrogram(tile, stats):
    """Apply the inverse of normalize_spectrogram.

    Parameters
    ----------
    tile : np.ndarray
        Normalized spectrogram tile.
    stats : tuple
        (mean, std) from normalize_spectrogram.

    Returns
    -------
    np.ndarray
        Denormalized tile.
    """
    mean, std = stats
    return tile * std + mean


def magnitude_to_audio(magnitude, phase=None, n_fft=512, hop_length=128):
    """Convert a magnitude spectrogram to audio using the Griffin-Lim algorithm.

    Parameters
    ----------
    magnitude : np.ndarray
        Magnitude spectrogram (shape: (freq_bins, time_frames)).
    phase : np.ndarray, optional
        Initial phase estimate. If None, use random phase.
    n_fft : int
        FFT size.
    hop_length : int
        Hop length between frames.

    Returns
    -------
    np.ndarray
        1D audio array.
    """
    if magnitude.ndim != 2:
        raise ValueError("Magnitude must be 2D")
    n_freq = magnitude.shape[0]
    # Ensure frequency bins match n_fft//2+1
    if n_freq != n_fft // 2 + 1:
        # Interpolate to match
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, n_freq)
        x_new = np.linspace(0, 1, n_fft // 2 + 1)
        magnitude = interp1d(x_old, magnitude, axis=0, kind='linear')(x_new)

    n_frames = magnitude.shape[1]
    if phase is None:
        rng = np.random.default_rng(0)
        phase = rng.uniform(0, 2 * np.pi, (n_fft // 2 + 1, n_frames))

    # Griffin-Lim iterations
    audio = None
    for _ in range(30):
        # Reconstruct complex spectrum
        complex_spec = magnitude * np.exp(1j * phase)
        # Inverse STFT
        audio = istft(complex_spec, hop_length=hop_length, n_fft=n_fft)
        # Forward STFT to get new phase
        new_spec = stft(audio, hop_length=hop_length, n_fft=n_fft)
        phase = np.angle(new_spec)

    return audio


def stft(signal, n_fft=512, hop_length=128):
    """Compute STFT of a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        1D audio signal.
    n_fft : int
        FFT size.
    hop_length : int
        Hop length.

    Returns
    -------
    np.ndarray
        Complex STFT of shape (n_fft//2+1, n_frames).
    """
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    n_samples = len(signal)
    # Pad to ensure at least one frame
    if n_samples < n_fft:
        signal = np.pad(signal, (0, n_fft - n_samples))
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    # Ensure enough length for n_frames
    if len(signal) < (n_frames - 1) * hop_length + n_fft:
        pad_len = (n_frames - 1) * hop_length + n_fft - len(signal)
        signal = np.pad(signal, (0, pad_len))
    windows = np.lib.stride_tricks.sliding_window_view(signal, n_fft)[::hop_length]
    # Apply Hann window
    window = np.hanning(n_fft)
    windows = windows * window
    spectrum = np.fft.rfft(windows, n=n_fft, axis=1)
    return spectrum.T


def istft(spectrum, hop_length=128, n_fft=512):
    """Compute inverse STFT.

    Parameters
    ----------
    spectrum : np.ndarray
        Complex STFT of shape (n_fft//2+1, n_frames).
    hop_length : int
        Hop length.
    n_fft : int
        FFT size.

    Returns
    -------
    np.ndarray
        1D audio signal.
    """
    if spectrum.ndim != 2:
        raise ValueError("Spectrum must be 2D")
    n_freq, n_frames = spectrum.shape
    if n_freq != n_fft // 2 + 1:
        raise ValueError(f"Expected {n_fft//2+1} frequency bins, got {n_freq}")
    window = np.hanning(n_fft)
    signal_len = (n_frames - 1) * hop_length + n_fft
    signal = np.zeros(signal_len)
    window_sum = np.zeros(signal_len)
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        frame = np.fft.irfft(spectrum[:, i], n=n_fft)
        signal[start:end] += frame * window
        window_sum[start:end] += window * window
    # Normalize by window overlap
    nonzero = window_sum > 1e-8
    signal[nonzero] /= window_sum[nonzero]
    return signal
