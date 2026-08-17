import numpy as np


class SpectralContrast:
    """Compute spectral contrast features from a spectrogram.

    Spectral contrast measures the difference between peaks and valleys in
    each frequency sub-band. It is commonly used for timbre and texture
    analysis. This module provides a function to compute spectral contrast
    for each time frame, which can be used to modulate diffusion parameters
    or to analyze the generated audio in real time.

    Parameters
    ----------
    n_bands : int, optional
        Number of sub-bands to divide the frequency range into.
    fmin : float, optional
        Minimum frequency in Hz for the first band.
    fmax : float, optional
        Maximum frequency in Hz for the last band.
    sample_rate : int, optional
        Sample rate of the audio, used for frequency bin calculation.
    """

    def __init__(self, n_bands=6, fmin=0.0, fmax=None, sample_rate=22050):
        self.n_bands = n_bands
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2
        self.sample_rate = sample_rate

    def compute(self, spectrogram):
        """Compute spectral contrast for each frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, n_frames), where each column is a
            magnitude spectrum (linear or dB).

        Returns
        -------
        np.ndarray
            2D array of shape (n_bands, n_frames) with spectral contrast values
            per band per frame.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")

        n_freq, n_frames = spectrogram.shape
        freq_bins = np.linspace(0, self.sample_rate / 2, n_freq)

        # Define band edges in Hz
        band_edges = np.linspace(self.fmin, self.fmax, self.n_bands + 1)

        # Compute contrast per frame per band
        contrast = np.zeros((self.n_bands, n_frames), dtype=np.float32)
        for band in range(self.n_bands):
            lo = band_edges[band]
            hi = band_edges[band + 1]
            # Find frequency bins in this band
            mask = (freq_bins >= lo) & (freq_bins <= hi)
            if not np.any(mask):
                continue
            band_spectrum = spectrogram[mask, :]  # shape (n_bins_in_band, n_frames)
            # Peak = max, valley = min along frequency axis
            peaks = np.max(band_spectrum, axis=0)
            valleys = np.min(band_spectrum, axis=0)
            # Contrast = peak - valley (with small epsilon to avoid zero)
            contrast[band, :] = peaks - valleys

        return contrast

    def normalize(self, contrast, eps=1e-8):
        """Normalize contrast values to [0, 1] per band.

        Parameters
        ----------
        contrast : np.ndarray
            Output from :meth:`compute`.
        eps : float, optional
            Small value to avoid division by zero.

        Returns
        -------
        np.ndarray
            Normalized contrast values.
        """
        contrast = np.asarray(contrast, dtype=np.float32)
        if contrast.ndim != 2:
            raise ValueError("Contrast must be 2D")
        min_val = np.min(contrast, axis=1, keepdims=True)
        max_val = np.max(contrast, axis=1, keepdims=True)
        denom = max_val - min_val + eps
        return (contrast - min_val) / denom
