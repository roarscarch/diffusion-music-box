import numpy as np


class SpectralContrast:
    """Compute spectral contrast features from a spectrogram.

    Spectral contrast measures the difference in amplitude between peaks
    and valleys in each frequency sub-band. It provides a harmonic-to-noise
    ratio estimation, useful for analyzing timbral richness and brightness.

    Parameters
    ----------
    n_bands : int, optional
        Number of sub-bands to divide the spectrum into. Default is 6.
    fmin : float, optional
        Minimum frequency in Hz for the bands. Default is 200.0.
    fmax : float, optional
        Maximum frequency in Hz for the bands. Default is 8000.0.
    """

    def __init__(self, n_bands=6, fmin=200.0, fmax=8000.0):
        self.n_bands = n_bands
        self.fmin = fmin
        self.fmax = fmax

    def _band_edges(self, sample_rate, n_freq_bins):
        """Compute frequency bin indices for sub-band edges.

        Returns
        -------
        list of tuple
            Each tuple contains (start_bin, end_bin) inclusive.
        """
        # Convert frequency bounds to bin indices
        max_freq = sample_rate / 2.0
        fmax = min(self.fmax, max_freq)
        if fmax <= self.fmin:
            fmax = max_freq

        # Use logarithmic spacing for bands
        log_min = np.log10(max(self.fmin, 1.0))
        log_max = np.log10(max(fmax, self.fmin + 1.0))
        edges = np.logspace(log_min, log_max, self.n_bands + 1)

        bands = []
        for i in range(self.n_bands):
            start_freq = edges[i]
            end_freq = edges[i + 1]
            start_bin = int(round(start_freq * n_freq_bins / max_freq))
            end_bin = int(round(end_freq * n_freq_bins / max_freq))
            # Ensure at least one bin per band
            if end_bin <= start_bin:
                end_bin = start_bin + 1
            # Clamp to valid range
            start_bin = max(0, start_bin)
            end_bin = min(n_freq_bins - 1, end_bin)
            bands.append((start_bin, end_bin))
        return bands

    def compute(self, spectrogram, sample_rate):
        """Compute spectral contrast for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (n_freq_bins, n_frames) representing magnitude
            or power spectrogram.
        sample_rate : int
            Sample rate of the audio.

        Returns
        -------
        np.ndarray
            2D array of shape (n_bands, n_frames) where each row contains
            the spectral contrast values for a sub-band.
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        n_freq_bins, n_frames = spectrogram.shape
        bands = self._band_edges(sample_rate, n_freq_bins)

        contrast = np.zeros((self.n_bands, n_frames), dtype=np.float32)
        for band_idx, (start, end) in enumerate(bands):
            if start >= end:
                continue
            band_data = spectrogram[start:end + 1, :]  # shape (band_bins, n_frames)
            if band_data.size == 0:
                continue
            # Compute peak and valley values per frame
            # Using percentile for robustness
            peak = np.percentile(band_data, 90, axis=0)
            valley = np.percentile(band_data, 10, axis=0)
            # Avoid division by zero
            denom = valley + 1e-10
            contrast[band_idx, :] = (peak - valley) / denom
        return contrast

    def compute_mean(self, spectrogram, sample_rate):
        """Compute mean spectral contrast across all bands for each frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (n_freq_bins, n_frames).
        sample_rate : int
            Sample rate of the audio.

        Returns
        -------
        np.ndarray
            1D array of length n_frames with mean contrast values.
        """
        contrast = self.compute(spectrogram, sample_rate)
        return np.mean(contrast, axis=0)
