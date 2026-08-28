import numpy as np


class SpectralFlux:
    """Compute spectral flux for onset and change detection.

    Spectral flux measures the magnitude of change in the spectrum between
    consecutive frames. It is commonly used for onset detection and for
    identifying moments of significant spectral change in music.

    Parameters
    ----------
    window : str, optional
        Window function to apply to the flux values: 'hann', 'hamming', or None.
    epsilon : float, optional
        Small constant to avoid division by zero in normalization.
    """

    def __init__(self, window=None, epsilon=1e-10):
        self.window = window
        self.epsilon = epsilon

    def compute(self, spectrogram):
        """Compute spectral flux for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitudes.
            Typically the magnitude spectrogram (non-negative).

        Returns
        -------
        np.ndarray
            1D array of length (time_frames - 1) with flux values per frame.
        """
        # Ensure input is at least 2D
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")

        # Compute difference between consecutive frames
        diff = np.diff(spectrogram, axis=1)
        # Half-wave rectification: only positive changes count
        flux = np.maximum(diff, 0)

        # Sum over frequency bins
        flux_per_frame = np.sum(flux, axis=0)

        # Apply window if specified
        if self.window is not None:
            if self.window == 'hann':
                win = np.hanning(len(flux_per_frame))
            elif self.window == 'hamming':
                win = np.hamming(len(flux_per_frame))
            else:
                raise ValueError(f"Unknown window type: {self.window}