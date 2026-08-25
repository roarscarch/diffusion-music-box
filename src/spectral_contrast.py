import numpy as np


class SpectralContrast:
    """Compute spectral contrast features from a spectrogram.

    Spectral contrast characterizes the difference between peaks and valleys
    in each frequency sub-band. It is useful for analyzing the timbral
    richness of the generated ambient music, helping to shape the diffusion
    output toward desired spectral textures.

    Parameters
    ----------
    n_bands : int, optional
        Number of sub-bands to divide the spectrum into.
    fmin : float, optional
        Minimum frequency in Hz for the lowest band.
    fmax : float, optional
        Maximum frequency in Hz for the highest band.
    sample_rate : int, optional
        Sample rate of the audio (used to map frequency bins).
    fft_size : int, optional
        FFT size used for the spectrogram (number of frequency bins).
    """

    def __init__(self, n_bands=6, fmin=50.0, fmax=8000.0, sample_rate=22050, fft_size=1024):
        self.n_bands = n_bands
        self.fmin = fmin
        self.fmax = fmax
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.freq_bins = fft_size // 2 + 1

        # Precompute band edges in Hz and bin indices
        band_edges_hz = np.geomspace(fmin, fmax, n_bands + 1)
        self.band_edges_bins = np.clip(
            np.round(band_edges_hz * fft_size / sample_rate).astype(int),
            0,
            self.freq_bins - 1
        )

    def compute(self, spectrogram):
        """Compute spectral contrast for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames), magnitude or power.

        Returns
        -------
        np.ndarray
            2D array of shape (n_bands, time_frames) containing contrast values
            (peak minus valley) for each band.
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if spectrogram.shape[0] != self.freq_bins:
            raise ValueError(
                f"Expected {self.freq_bins} frequency bins, got {spectrogram.shape[0]}"
            )

        n_frames = spectrogram.shape[1]
        contrast = np.zeros((self.n_bands, n_frames), dtype=np.float32)

        for band_idx in range(self.n_bands):
            start = self.band_edges_bins[band_idx]
            end = self.band_edges_bins[band_idx + 1]
            if end <= start:
                continue
            band = spectrogram[start:end, :]
            # Use log magnitude to reduce dynamic range
            log_band = np.log(band + 1e-10)
            peak = np.max(log_band, axis=0)
            valley = np.min(log_band, axis=0)
            contrast[band_idx, :] = peak - valley

        return contrast

    def compute_mean(self, spectrogram):
        """Compute the mean spectral contrast over time.

        Useful for summarizing the overall spectral texture of a segment.

        Returns
        -------
        np.ndarray
            1D array of length n_bands.
        """
        contrast = self.compute(spectrogram)
        return np.mean(contrast, axis=1)
