import numpy as np


class SpectralKurtosis:
    """Compute the spectral kurtosis of a spectrogram.

    Spectral kurtosis measures the peakedness or heaviness of the tails of the
    frequency distribution at each time frame. It is useful for identifying
    transient, impulsive sounds (high kurtosis) versus steady, tonal sounds
    (lower kurtosis). This module provides a function to compute the kurtosis
    across frequency bins for each frame, returning a time series.

    Parameters
    ----------
    eps : float, optional
        Small constant to avoid division by zero.
    """

    def __init__(self, eps=1e-10):
        self.eps = eps

    def compute(self, spectrogram):
        """Compute spectral kurtosis for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitude
            spectrogram values (non-negative).

        Returns
        -------
        np.ndarray
            1D array of length time_frames with kurtosis values.
            Kurtosis is defined as the fourth standardized moment minus 3
            (excess kurtosis). Positive values indicate a more peaked
            distribution than a Gaussian, negative values a flatter one.
        """
        spec = np.asarray(spectrogram, dtype=np.float64)
        if spec.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if np.any(spec < 0):
            raise ValueError("Spectrogram must be non-negative")

        # Normalize each frame to sum to 1 (probability distribution)
        frame_sums = spec.sum(axis=0, keepdims=True)
        # Avoid division by zero: if a frame has zero energy, set kurtosis to 0
        zero_frames = frame_sums[0] < self.eps
        probs = spec / (frame_sums + self.eps)

        # Compute mean frequency (first moment)
        freq_indices = np.arange(spec.shape[0], dtype=np.float64)[:, None]
        mean = np.sum(probs * freq_indices, axis=0)

        # Center around mean
        centered = freq_indices - mean[None, :]

        # Compute second and fourth moments
        variance = np.sum(probs * centered**2, axis=0)
        fourth_moment = np.sum(probs * centered**4, axis=0)

        # Kurtosis = fourth_moment / variance^2 - 3
        kurtosis = np.zeros_like(variance)
        valid = variance > self.eps
        kurtosis[valid] = fourth_moment[valid] / (variance[valid]**2 + self.eps) - 3.0
        kurtosis[zero_frames] = 0.0

        return kurtosis
