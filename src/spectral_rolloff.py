import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of an audio signal.

    Spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is contained. It is a
    measure of the brightness or high-frequency content of a signal and is
    commonly used in music information retrieval for genre classification
    and timbral analysis.

    This module provides both a function and a class interface for
    computing spectral rolloff from either a time-domain signal or a
    precomputed magnitude spectrogram.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz.
    rolloff_percent : float, optional
        Percentage of total spectral energy to consider (0.0 to 1.0).
        Default is 0.85 (85%).
    """

    def __init__(self, sample_rate=22050, rolloff_percent=0.85):
        self.sample_rate = sample_rate
        self.rolloff_percent = rolloff_percent

    def _compute_rolloff(self, magnitude):
        """Compute spectral rolloff for a single magnitude spectrum.

        Parameters
        ----------
        magnitude : np.ndarray
            1D array of magnitude values for a single frame.

        Returns
        -------
        float
            Rolloff frequency in Hz.
        """
        if magnitude.ndim != 1:
            raise ValueError("Magnitude must be 1D")
        if magnitude.size == 0:
            return 0.0
        total_energy = np.sum(magnitude)
        if total_energy == 0:
            return 0.0
        cumulative = np.cumsum(magnitude)
        threshold = self.rolloff_percent * total_energy
        # Find the first bin where cumulative energy exceeds threshold
        idx = np.searchsorted(cumulative, threshold)
        if idx >= len(magnitude):
            idx = len(magnitude) - 1
        # Convert bin index to frequency (Hz)
        # FFT bin resolution = sample_rate / n_fft, but here we assume
        # magnitude length corresponds to n_fft/2 + 1 with linear spacing
        # from 0 to Nyquist (sample_rate/2).
        n_bins = len(magnitude)
        freq = idx * (self.sample_rate / 2) / (n_bins - 1)
        return freq

    def from_spectrogram(self, spectrogram):
        """Compute rolloff for each frame of a magnitude spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (n_frames, n_freq_bins) containing magnitude
            values. Frames are along axis 0, frequency bins along axis 1.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies (Hz) for each frame.
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D with shape (n_frames, n_freq_bins)")
        rolloffs = np.apply_along_axis(self._compute_rolloff, axis=1, arr=spectrogram)
        return rolloffs

    def from_signal(self, signal, fft_size=1024, hop_length=256):
        """Compute spectral rolloff from a time-domain signal.

        Parameters
        ----------
        signal : np.ndarray
            1D audio signal.
        fft_size : int, optional
            FFT size for the spectrogram.
        hop_length : int, optional
            Hop length between frames.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies (Hz) for each frame.
        """
        # Compute STFT magnitude using numpy (no external libs)
        n_frames = 1 + (len(signal) - fft_size) // hop_length
        if n_frames <= 0:
            return np.array([], dtype=np.float32)
        spectrogram = np.zeros((n_frames, fft_size // 2 + 1), dtype=np.float32)
        window = np.hanning(fft_size)
        for i in range(n_frames):
            start = i * hop_length
            frame = signal[start:start + fft_size] * window
            spectrum = np.fft.rfft(frame)
            spectrogram[i] = np.abs(spectrum)
        return self.from_spectrogram(spectrogram)

    def compute(self, spectrogram):
        """Alias for from_spectrogram."""
        return self.from_spectrogram(spectrogram)


def compute_spectral_rolloff(spectrogram, sample_rate=22050, rolloff_percent=0.85):
    """Compute spectral rolloff for a magnitude spectrogram.

    Convenience function that creates a SpectralRolloff instance and calls
    from_spectrogram.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (n_frames, n_freq_bins).
    sample_rate : int
        Sample rate in Hz.
    rolloff_percent : float, optional
        Rolloff percentage (default 0.85).

    Returns
    -------
    np.ndarray
        1D array of rolloff frequencies.
    """
    rolloff = SpectralRolloff(sample_rate=sample_rate, rolloff_percent=rolloff_percent)
    return rolloff.from_spectrogram(spectrogram)
