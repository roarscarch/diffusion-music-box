import numpy as np


class SpectralCentroid:
    """Compute the spectral centroid of a signal or spectrogram.

    The spectral centroid is a measure of the 'brightness' of a sound,
    defined as the weighted mean of the frequencies present in the signal,
    weighted by their magnitudes. This module provides functionality to
    compute the centroid from raw audio (via STFT) or from a precomputed
    spectrogram, and to smooth the values over time for stable analysis.
    Useful for real-time timbre analysis and interactive control.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio in Hz.
    fft_size : int, optional
        FFT size for spectrogram computation (if not provided directly).
    hop_length : int, optional
        Hop length in samples between frames.
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.freq_bins = fft_size // 2 + 1

    def _stft(self, audio):
        """Compute the magnitude spectrogram of an audio signal.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            Magnitude spectrogram of shape (freq_bins, n_frames).
        """
        # Pad audio to ensure at least one frame
        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)))

        n_frames = 1 + (len(audio) - self.fft_size) // self.hop_length
        frames = np.zeros((n_frames, self.fft_size))
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = audio[start:start + self.fft_size]
        window = np.hanning(self.fft_size)
        frames *= window
        spectrum = np.fft.rfft(frames, axis=1)
        return np.abs(spectrum).T

    def from_audio(self, audio):
        """Compute spectral centroid from raw audio.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            Spectral centroid values per frame, shape (n_frames,).
        """
        spectrogram = self._stft(audio)
        return self.from_spectrogram(spectrogram)

    def from_spectrogram(self, spectrogram):
        """Compute spectral centroid from a magnitude spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D float array of shape (freq_bins, n_frames).

        Returns
        -------
        np.ndarray
            Spectral centroid values per frame, shape (n_frames,).
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)
        # Ensure frequency array matches spectrogram row count
        if len(freqs) != spectrogram.shape[0]:
            # Recompute based on actual shape
            freqs = np.linspace(0, self.sample_rate / 2, spectrogram.shape[0])
        # Weighted mean of frequencies
        total_energy = np.sum(spectrogram, axis=0)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            centroid = np.sum(spectrogram * freqs[:, np.newaxis], axis=0) / total_energy
        # Replace NaN (silent frames) with 0
        centroid = np.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0)
        return centroid

    def smooth(self, values, alpha=0.1):
        """Apply exponential moving average smoothing to centroid values.

        Parameters
        ----------
        values : np.ndarray
            Array of centroid values.
        alpha : float, optional
            Smoothing factor between 0 and 1. Higher values give more weight to recent samples.

        Returns
        -------
        np.ndarray
            Smoothed array of same shape.
        """
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in (0, 1]")
        smoothed = np.zeros_like(values, dtype=np.float64)
        if len(values) == 0:
            return smoothed
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed
