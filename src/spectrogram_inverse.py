import numpy as np


class SpectrogramInverter:
    """Invert a magnitude spectrogram back to audio using the Griffin-Lim algorithm.

    This module reconstructs an audio signal from a magnitude spectrogram
    (frequency-time image) by iteratively estimating the phase and applying
    the inverse short-time Fourier transform (ISTFT). It is essential for
    converting the diffusion-generated spectrogram tiles into playable audio.
    The implementation supports both standard Griffin-Lim and a faster
    accelerated variant (with momentum) for real-time use.

    Parameters
    ----------
    n_fft : int
        FFT size used for the spectrogram (number of frequency bins = n_fft // 2 + 1).
    hop_length : int
        Number of samples between successive frames.
    window : str or np.ndarray, optional
        Window function to apply during STFT. Can be a string name (e.g., 'hann')
        or a 1D array of length n_fft. Default 'hann'.
    momentum : float, optional
        Momentum factor for accelerated Griffin-Lim (0 to 1). 0 means standard
        Griffin-Lim, higher values speed convergence. Default 0.99.
    """

    def __init__(self, n_fft=1024, hop_length=256, window='hann', momentum=0.99):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.momentum = momentum
        self.freq_bins = n_fft // 2 + 1

        if isinstance(window, str):
            self.window = self._get_window(window)
        else:
            self.window = np.asarray(window, dtype=np.float32)
            if self.window.shape[0] != n_fft:
                raise ValueError(f"Window length {self.window.shape[0]} does not match n_fft {n_fft}")

        # Precompute window normalization for overlap-add
        self._window_sum = self._compute_window_sum()

    def _get_window(self, name):
        """Return a window function array of length n_fft."""
        if name == 'hann':
            return np.hanning(self.n_fft).astype(np.float32)
        elif name == 'hamming':
            return np.hamming(self.n_fft).astype(np.float32)
        elif name == 'blackman':
            return np.blackman(self.n_fft).astype(np.float32)
        else:
            raise ValueError(f"Unsupported window: {name}")

    def _compute_window_sum(self):
        """Compute the sum of squared window values for normalization."""
        # For overlap-add, the denominator is sum of window^2 over hops
        # We'll compute it for a signal of length n_fft + hop_length * (something)
        # but for simplicity, use the standard formula for periodic windows.
        # In practice, we normalize by the window sum at each frame.
        return np.sum(self.window ** 2)

    def _stft(self, audio):
        """Compute short-time Fourier transform (STFT) of audio.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            Complex STFT matrix of shape (freq_bins, n_frames).
        """
        n_samples = len(audio)
        n_frames = 1 + (n_samples - self.n_fft) // self.hop_length
        if n_frames <= 0:
            raise ValueError("Audio too short for given FFT size")
        # Pad audio to ensure full frames
        padded_len = (n_frames - 1) * self.hop_length + self.n_fft
        padded = np.zeros(padded_len, dtype=np.float32)
        padded[:n_samples] = audio
        stft = np.zeros((self.freq_bins, n_frames), dtype=np.complex64)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = padded[start:start + self.n_fft] * self.window
            stft[:, i] = np.fft.rfft(frame)
        return stft

    def _istft(self, stft):
        """Compute inverse STFT (ISTFT) from a complex spectrogram.

        Parameters
        ----------
        stft : np.ndarray
            Complex STFT matrix of shape (freq_bins, n_frames).

        Returns
        -------
        np.ndarray
            1D float array of reconstructed audio samples.
        """
        n_frames = stft.shape[1]
        # Compute output length
        n_samples = (n_frames - 1) * self.hop_length + self.n_fft
        audio = np.zeros(n_samples, dtype=np.float32)
        window_sum = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = np.fft.irfft(stft[:, i], n=self.n_fft)
            # Apply window (analysis window is same as synthesis for Griffin-Lim)
            frame = frame * self.window
            audio[start:start + self.n_fft] += frame
            window_sum[start:start + self.n_fft] += self.window ** 2
        # Normalize by window sum, avoiding division by zero
        eps = 1e-10
        audio = audio / (window_sum + eps)
        return audio.astype(np.float32)

    def invert(self, magnitude, iterations=32, initial_phase=None, return_phase=False):
        """Reconstruct audio from a magnitude spectrogram.

        Parameters
        ----------
        magnitude : np.ndarray
            2D array of shape (freq_bins, n_frames) containing non-negative magnitudes.
        iterations : int, optional
            Number of iterations for phase reconstruction. Default 32.
        initial_phase : np.ndarray, optional
            Initial phase estimate (complex array) to start from. If None, uses random
            phase. Default None.
        return_phase : bool, optional
            If True, also return the final phase estimate. Default False.

        Returns
        -------
        np.ndarray or tuple
            Reconstructed audio as 1D float array. If return_phase is True, returns
            (audio, phase) where phase is the final complex spectrogram.
        """
        magnitude = np.asarray(magnitude, dtype=np.float32)
        if magnitude.ndim != 2:
            raise ValueError("Magnitude must be 2D")
        if magnitude.shape[0] != self.freq_bins:
            raise ValueError(f"Magnitude shape {magnitude.shape[0]} does not match freq_bins {self.freq_bins}