import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of a spectrogram.

    Spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is contained. It is a
    measure of the brightness or sharpness of a sound and is commonly used
    in audio analysis for timbre characterization.

    This module provides a function to compute the rolloff frequency for
    each time frame of a spectrogram, which can be used to analyze the
    evolving spectral characteristics of generated ambient music.
    """

    def __init__(self, sample_rate=22050, percentage=0.85):
        """Initialize the spectral rolloff calculator.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the audio signal in Hz.
        percentage : float, optional
            Percentage of total energy to consider (default 0.85).
        """
        if not 0.0 < percentage <= 1.0:
            raise ValueError("Percentage must be in (0, 1]")
        self.sample_rate = sample_rate
        self.percentage = percentage

    def compute(self, spectrogram):
        """Compute the spectral rolloff for each frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitude
            spectrogram values (non-negative).

        Returns
        -------
        np.ndarray
            1D array of length time_frames with the rolloff frequency in Hz
            for each frame.
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if np.any(spectrogram < 0):
            raise ValueError("Spectrogram values must be non-negative")

        freq_bins, num_frames = spectrogram.shape
        # Compute cumulative sum along frequency axis
        cumsum = np.cumsum(spectrogram, axis=0)
        total_energy = cumsum[-1, :]
        # Avoid division by zero
        total_energy = np.where(total_energy == 0, 1e-12, total_energy)
        # Find the first bin where cumulative energy exceeds the threshold
        threshold = self.percentage * total_energy
        # Use argmax to find first True index (works because cumsum is monotonic)
        # We need to handle the case where no bin exceeds threshold (shouldn't happen)
        rolloff_bins = np.argmax(cumsum >= threshold, axis=0)
        # Convert bin index to frequency
        # Frequency resolution = sample_rate / (2 * (freq_bins - 1)) for one-sided spectrum
        # But we don't know FFT size, so we approximate using linear mapping from 0 to Nyquist
        nyquist = self.sample_rate / 2.0
        return rolloff_bins * (nyquist / (freq_bins - 1))
