import numpy as np


class SpectralFlux:
    """Compute spectral flux for onset detection and energy variation analysis.

    Spectral flux measures the change in magnitude spectrum between consecutive
    frames. High flux values indicate transients or onsets, which can be used
    to detect rhythmic events in the ambient music or to trigger changes in
    the diffusion parameters.

    Parameters
    ----------
    window_size : int, optional
        Number of frames to average for smoothing the flux curve.
    """

    def __init__(self, window_size=5):
        self.window_size = max(1, int(window_size))

    def compute(self, spectrogram):
        """Compute spectral flux from a magnitude spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitude
            values. Should be non-negative.

        Returns
        -------
        np.ndarray
            1D array of length time_frames-1 representing the spectral flux
            between consecutive frames.
        """
        spec = np.asarray(spectrogram, dtype=np.float32)
        if spec.ndim != 2:
            raise ValueError("Spectrogram must be 2D")

        # Compute magnitude difference between consecutive frames
        diff = np.diff(spec, axis=1)
        # Only consider positive changes (increase in energy)
        flux = np.maximum(diff, 0).sum(axis=0)

        # Smooth the flux curve with a moving average
        if self.window_size > 1 and len(flux) > 0:
            kernel = np.ones(self.window_size, dtype=np.float32) / self.window_size
            flux = np.convolve(flux, kernel, mode='same')

        return flux

    def detect_onsets(self, spectrogram, threshold=0.5, min_gap=3):
        """Detect onset frames based on spectral flux.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D magnitude spectrogram.
        threshold : float, optional
            Relative threshold (0.0 to 1.0) compared to the maximum flux.
            Frames with flux above this threshold are considered onsets.
        min_gap : int, optional
            Minimum number of frames between onsets to avoid duplicates.

        Returns
        -------
        list of int
            Frame indices where onsets occur.
        """
        flux = self.compute(spectrogram)
        if len(flux) == 0:
            return []

        max_flux = np.max(flux)
        if max_flux == 0:
            return []

        # Find frames above threshold
        threshold_val = threshold * max_flux
        candidates = np.where(flux > threshold_val)[0]

        # Apply minimum gap between onsets
        onsets = []
        last_onset = -min_gap - 1
        for idx in candidates:
            if idx - last_onset >= min_gap:
                onsets.append(int(idx))
                last_onset = idx

        return onsets
