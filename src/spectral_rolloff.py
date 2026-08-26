import numpy as np


class SpectralRolloff:
    """Calculate the spectral rolloff frequency of an audio signal.

    The spectral rolloff is the frequency below which a specified percentage
    (typically 85% or 95%) of the total spectral energy is contained. It is a
    simple measure of spectral shape that can be used to characterize the
    brightness or timbre of a sound. This module provides a function to
    compute the rolloff for a given magnitude spectrum or a full spectrogram.
    """

    def __init__(self, sample_rate=22050, percentage=0.85):
        """Initialize the spectral rolloff calculator.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the audio signal in Hz (used to convert bin indices
            to frequencies).
        percentage : float, optional
            Fraction of total energy (0.0 to 1.0) below which the rolloff
            frequency is defined. Common values are 0.85 and 0.95.
        """
        if not 0.0 < percentage < 1.0:
            raise ValueError("percentage must be between 0 and 1")
        self.sample_rate = sample_rate
        self.percentage = percentage

    def compute(self, spectrum):
        """Compute the spectral rolloff frequency for a single magnitude spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            1D array of magnitude spectrum values (non-negative).

        Returns
        -------
        float
            The rolloff frequency in Hz.
        """
        spectrum = np.asarray(spectrum, dtype=np.float64)
        if spectrum.ndim != 1:
            raise ValueError("spectrum must be 1D")
        total_energy = np.sum(spectrum)
        if total_energy <= 0:
            return 0.0
        cumulative = np.cumsum(spectrum)
        threshold = total_energy * self.percentage
        # Find the first bin where cumulative energy exceeds threshold
        idx = np.searchsorted(cumulative, threshold)
        idx = min(idx, len(spectrum) - 1)
        # Convert bin index to frequency
        return idx * self.sample_rate / (2 * (len(spectrum) - 1)) if len(spectrum) > 1 else 0.0

    def compute_spectrogram(self, spectrogram):
        """Compute spectral rolloff for each frame of a spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) with magnitude values.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies (in Hz) for each time frame.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float64)
        if spectrogram.ndim != 2:
            raise ValueError("spectrogram must be 2D")
        n_frames = spectrogram.shape[1]
        rolloffs = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            rolloffs[i] = self.compute(spectrogram[:, i])
        return rolloffs
