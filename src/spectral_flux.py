import numpy as np


class SpectralFlux:
    """Compute the spectral flux of a spectrogram.

    Spectral flux measures the rate of change in the magnitude spectrum
    between consecutive frames. It is commonly used for onset detection
    and to identify significant spectral changes in audio signals. This
    module provides a clean, reusable implementation for the diffusion
    music box project, enabling real-time analysis and parameter
    modulation based on spectral activity.

    Parameters
    ----------
    hop_length : int
        Hop length in samples between frames (used for normalization).
    """

    def __init__(self, hop_length=256):
        self.hop_length = hop_length

    def compute(self, spectrogram):
        """Compute spectral flux across time frames.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing
            magnitude values (non-negative).

        Returns
        -------
        np.ndarray
            1D array of length (time_frames - 1) with the flux value
            between each consecutive pair of frames.

        Raises
        ------
        ValueError
            If the spectrogram is not 2D or contains negative values.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if np.any(spectrogram < 0):
            raise ValueError("Spectrogram must contain non-negative values")

        # Compute difference between consecutive frames, keep positive changes
        diff = np.diff(spectrogram, axis=1)
        flux = np.sum(np.maximum(diff, 0), axis=0)

        # Normalize by hop length to get flux per second
        flux = flux / self.hop_length
        return flux

    def normalize(self, flux, epsilon=1e-8):
        """Normalize flux values to a [0, 1] range.

        Parameters
        ----------
        flux : np.ndarray
            1D array of flux values.
        epsilon : float, optional
            Small constant to avoid division by zero.

        Returns
        -------
        np.ndarray
            Normalized flux values.
        """
        flux = np.asarray(flux, dtype=np.float32)
        max_val = np.max(flux)
        if max_val < epsilon:
            return np.zeros_like(flux)
        return flux / max_val

    def detect_onsets(self, flux, threshold=0.5, min_distance=1):
        """Detect onset frames based on spectral flux peaks.

        Parameters
        ----------
        flux : np.ndarray
            1D array of flux values.
        threshold : float, optional
            Normalized threshold (0.0 to 1.0) for peak detection.
        min_distance : int, optional
            Minimum number of frames between onsets.

        Returns
        -------
        list of int
            Frame indices where onsets are detected.
        """
        flux = np.asarray(flux, dtype=np.float32)
        if flux.ndim != 1:
            raise ValueError("Flux must be 1D")
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        normalized = self.normalize(flux)
        onsets = []
        last_onset = -min_distance
        for i, val in enumerate(normalized):
            if val >= threshold and (i - last_onset) >= min_distance:
                onsets.append(i)
                last_onset = i
        return onsets

    def compute_adaptive_threshold(self, flux, window=100, multiplier=1.5):
        """Compute an adaptive threshold for onset detection.

        Parameters
        ----------
        flux : np.ndarray
            1D array of flux values.
        window : int, optional
            Number of frames to consider for the moving average.
        multiplier : float, optional
            Multiplier for the moving average to set the threshold.

        Returns
        -------
        np.ndarray
            Array of threshold values per frame (length matches flux).
        """
        flux = np.asarray(flux, dtype=np.float32)
        if flux.ndim != 1:
            raise ValueError("Flux must be 1D")
        if window <= 0:
            raise ValueError("Window must be positive")

        # Compute moving average of flux
        kernel = np.ones(window) / window
        moving_avg = np.convolve(flux, kernel, mode='same')
        threshold = moving_avg * multiplier
        return threshold
