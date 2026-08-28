import numpy as np


class SpectralContrast:
    """Compute spectral contrast features from a spectrogram.

    Spectral contrast measures the difference in amplitude between spectral
    peaks and valleys in each sub-band. It is useful for analyzing timbral
    texture in ambient music, where percussive or noisy elements can be
    distinguished from sustained tones.

    Parameters
    ----------
    n_bands : int, optional
        Number of sub-bands to divide the spectrum into (default 6).
    """

    def __init__(self, n_bands=6):
        self.n_bands = n_bands

    def _split_bands(self, magnitude):
        """Split the magnitude spectrum into sub-bands.

        Parameters
        ----------
        magnitude : np.ndarray
            1D array of magnitude values (e.g., one frame of a spectrogram).

        Returns
        -------
        list of np.ndarray
            List of sub-band magnitude arrays.
        """
        n_freq = len(magnitude)
        # Divide the spectrum into equal-sized bands (or more naturally, octave bands)
        # For simplicity, use linear division
        band_edges = np.linspace(0, n_freq, self.n_bands + 1, dtype=int)
        bands = []
        for i in range(self.n_bands):
            start = band_edges[i]
            end = band_edges[i + 1]
            if start < end:
                bands.append(magnitude[start:end])
            else:
                bands.append(np.array([0.0]))
        return bands

    def _band_contrast(self, band):
        """Compute contrast for a single band.

        Parameters
        ----------
        band : np.ndarray
            Magnitude values in the band.

        Returns
        -------
        float
            Contrast value (peak - valley) in the band.
        """
        if len(band) == 0:
            return 0.0
        peak = np.max(band)
        valley = np.min(band)
        return peak - valley

    def compute(self, spectrogram):
        """Compute spectral contrast for each frame of a spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (n_frames, n_freq_bins) containing magnitude
            values (non-negative).

        Returns
        -------
        np.ndarray
            2D array of shape (n_frames, n_bands) with contrast values.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if np.any(spectrogram < 0):
            raise ValueError("Spectrogram must contain non-negative magnitudes")

        n_frames, n_freq = spectrogram.shape
        contrast = np.zeros((n_frames, self.n_bands), dtype=np.float32)

        for i in range(n_frames):
            frame = spectrogram[i]
            bands = self._split_bands(frame)
            for j, band in enumerate(bands):
                contrast[i, j] = self._band_contrast(band)

        return contrast

    def __call__(self, spectrogram):
        """Alias for compute() to allow callable usage."""
        return self.compute(spectrogram)
