import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of a signal.

    The spectral rolloff is the frequency below which a specified percentage
    (e.g., 85%) of the total spectral energy is contained. It is commonly
    used for audio analysis to characterize the brightness or timbre of a
    sound. This module provides a function to compute the rolloff for a
    given magnitude spectrum or a set of frames.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz.
    rolloff_percent : float, optional
        Percentage of total energy below the rolloff frequency (0.0 to 1.0).
        Default is 0.85 (85%).
    """

    def __init__(self, sample_rate=22050, rolloff_percent=0.85):
        if not 0.0 < rolloff_percent <= 1.0:
            raise ValueError("rolloff_percent must be in (0, 1]")
        self.sample_rate = sample_rate
        self.rolloff_percent = rolloff_percent

    def compute(self, magnitude_spectrum):
        """Compute spectral rolloff for a single magnitude spectrum.

        Parameters
        ----------
        magnitude_spectrum : np.ndarray
            1D array of magnitude values (non-negative) from an FFT.

        Returns
        -------
        float
            Rolloff frequency in Hz.
        """
        mag = np.asarray(magnitude_spectrum, dtype=np.float64)
        if mag.ndim != 1:
            raise ValueError("magnitude_spectrum must be 1D")
        if np.any(mag < 0):
            raise ValueError("Magnitude spectrum cannot contain negative values")

        total_energy = np.sum(mag)
        if total_energy <= 0:
            return 0.0

        cumulative = np.cumsum(mag)
        threshold = self.rolloff_percent * total_energy
        # Find the first index where cumulative energy exceeds threshold
        idx = np.searchsorted(cumulative, threshold)
        # Map bin index to frequency
        freq = idx * self.sample_rate / (2 * (len(mag) - 1)) if len(mag) > 1 else 0.0
        return float(freq)

    def compute_frames(self, magnitude_frames):
        """Compute spectral rolloff for each frame in a 2D array.

        Parameters
        ----------
        magnitude_frames : np.ndarray
            2D array of shape (n_frames, n_bins) containing magnitude spectra.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies in Hz for each frame.
        """
        frames = np.asarray(magnitude_frames, dtype=np.float64)
        if frames.ndim != 2:
            raise ValueError("magnitude_frames must be 2D")
        return np.array([self.compute(frame) for frame in frames])
