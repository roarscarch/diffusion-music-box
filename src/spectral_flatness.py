import numpy as np


class SpectralFlatness:
    """Compute spectral flatness (Wiener entropy) of a spectrogram.

    Spectral flatness measures how noise-like versus tone-like a signal is.
    It is the ratio of the geometric mean to the arithmetic mean of the power
    spectrum. Values range from 0 (pure tone) to 1 (white noise). This module
    is useful for analyzing and controlling the texture of the generated
    ambient music, allowing the system to adjust parameters based on the
    desired sonic character.

    Parameters
    ----------
    epsilon : float, optional
        Small constant to avoid division by zero and log of zero.
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def compute(self, spectrogram):
        """Compute spectral flatness for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitude
            or power values. The values should be non-negative.

        Returns
        -------
        np.ndarray
            1D array of length time_frames with spectral flatness values
            in the range [0, 1].

        Raises
        ------
        ValueError
            If the input is not a 2D array or contains negative values.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float64)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be a 2D array")
        if np.any(spectrogram < 0):
            raise ValueError("Spectrogram values must be non-negative")

        # Add epsilon to avoid zero values
        power = spectrogram + self.epsilon

        # Geometric mean = exp(mean(log(power))) across frequency bins
        log_mean = np.mean(np.log(power), axis=0)
        geometric_mean = np.exp(log_mean)

        # Arithmetic mean across frequency bins
        arithmetic_mean = np.mean(power, axis=0)

        # Spectral flatness = geometric mean / arithmetic mean
        flatness = geometric_mean / arithmetic_mean

        # Clip to [0, 1] for numerical safety
        flatness = np.clip(flatness, 0.0, 1.0)

        return flatness.astype(np.float32)

    def compute_global(self, spectrogram):
        """Compute a single scalar spectral flatness for the entire spectrogram.

        This is a convenience method that flattens all frequency bins and time
        frames into one distribution, which is useful for overall texture
        analysis.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames).

        Returns
        -------
        float
            Spectral flatness scalar in the range [0, 1].
        """
        flatness_per_frame = self.compute(spectrogram)
        return float(np.mean(flatness_per_frame))
