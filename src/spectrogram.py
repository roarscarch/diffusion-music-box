import numpy as np


def stft(x, n_fft=512, hop_length=128, window='hann'):
    """Short-time Fourier transform.

    Parameters
    ----------
    x : np.ndarray
        1D input signal.
    n_fft : int
        FFT size.
    hop_length : int
        Hop between frames.
    window : str
        Window type: 'hann' or 'hamming'.

    Returns
    -------
    np.ndarray
        Complex spectrogram of shape (n_freq, n_frames).
    """
    if window == 'hann':
        win = np.hanning(n_fft)
    elif window == 'hamming':
        win = np.hamming(n_fft)
    else:
        raise ValueError(f"Unsupported window: {window}")

    n_frames = 1 + (len(x) - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop_length][:n_frames]
    return np.fft.rfft(frames * win, axis=1).T


def istft(S, n_fft=512, hop_length=128, window='hann'):
    """Inverse short-time Fourier transform with overlap-add.

    Parameters
    ----------
    S : np.ndarray
        Complex spectrogram of shape (n_freq, n_frames).
    n_fft : int
        FFT size.
    hop_length : int
        Hop between frames.
    window : str
        Window type used during STFT.

    Returns
    -------
    np.ndarray
        Reconstructed 1D signal.
    """
    if window == 'hann':
        win = np.hanning(n_fft)
    elif window == 'hamming':
        win = np.hamming(n_fft)
    else:
        raise ValueError(f"Unsupported window: {window}")

    n_frames = S.shape[1]
    expected_len = (n_frames - 1) * hop_length + n_fft
    out = np.zeros(expected_len, dtype=np.float64)
    window_sum = np.zeros(expected_len, dtype=np.float64)

    # Analysis window squared for perfect reconstruction with overlap-add
    win_sq = win ** 2

    for i in range(n_frames):
        start = i * hop_length
        frame = np.fft.irfft(S[:, i], n=n_fft)
        out[start:start + n_fft] += frame * win
        window_sum[start:start + n_fft] += win_sq

    # Avoid division by zero (use small epsilon)
    window_sum[window_sum < 1e-10] = 1.0
    return out / window_sum


def spectrogram_to_audio(spectrogram_magnitude, phase=None, n_fft=512, hop_length=128):
    """Convert a magnitude spectrogram to audio using random or provided phase.

    Parameters
    ----------
    spectrogram_magnitude : np.ndarray
        Magnitude spectrogram of shape (n_freq, n_frames).
    phase : np.ndarray or None
        Optional phase spectrogram. If None, random phase is generated.
    n_fft : int
        FFT size.
    hop_length : int
        Hop between frames.

    Returns
    -------
    np.ndarray
        Reconstructed audio signal.
    """
    n_freq, n_frames = spectrogram_magnitude.shape
    if phase is None:
        # Random phase in [-pi, pi]
        phase = np.random.uniform(-np.pi, np.pi, size=(n_freq, n_frames))
    complex_spec = spectrogram_magnitude * np.exp(1j * phase)
    return istft(complex_spec, n_fft=n_fft, hop_length=hop_length)
