import numpy as np


class SpectralCentroid:
    """Compute the spectral centroid of a spectrogram tile.

    The spectral centroid is a measure of the center of mass of the
    spectrum, indicating the brightness or sharpness of the sound.
    This module provides a function to compute the centroid for each
    time frame of a spectrogram, which can be used for analysis or
    for controlling synthesis parameters in real time.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio (used to map frequency bins to Hz).
    fft_size : int
        FFT size used for the spectrogram (number of frequency bins).
    """

    def __init__(self, sample_rate=22050, fft_size=1024):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.freq_bins = fft_size // 2 + 1

    def compute(self, spectrogram):
        """Compute spectral centroid for each frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitude
            or power values. Frequencies are assumed to be linearly spaced
            from 0 to Nyquist.

        Returns
        -------
        np.ndarray
            1D array of length time_frames with centroid values in Hz.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if spectrogram.shape[0] != self.freq_bins:
            raise ValueError(
                f"Expected {self.freq_bins} frequency bins, got {spectrogram.shape[0]}"
            )

        # Frequency values for each bin in Hz
        freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)

        # Weighted mean of frequencies
        total_energy = np.sum(spectrogram, axis=0)
        # Avoid division by zero
        total_energy = np.maximum(total_energy, 1e-10)
        centroid = np.sum(spectrogram * freqs[:, np.newaxis], axis=0) / total_energy

        return centroid
