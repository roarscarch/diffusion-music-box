import numpy as np


class SpectralContrast:
    """Compute spectral contrast features from a spectrogram.

    Spectral contrast describes the difference between peaks and valleys in
    the spectrum, which can be used to characterize the texture of ambient
    sound. This module splits the frequency range into sub-bands and computes
    the spectral peak, valley, and contrast for each band, providing a
    compact feature vector that can drive parameter modulation.

    Parameters
    ----------
    n_bands : int, optional
        Number of sub-bands to split the spectrum into.
    fmin : float, optional
        Minimum frequency in Hz for band splitting.
    fmax : float, optional
        Maximum frequency in Hz for band splitting.
    sample_rate : int, optional
        Sample rate of the audio (used to map frequency bins).
    fft_size : int, optional
        FFT size used for the spectrogram.
    """

    def __init__(self, n_bands=6, fmin=50.0, fmax=11025.0, sample_rate=22050, fft_size=1024):
        self.n_bands = n_bands
        self.fmin = fmin
        self.fmax = fmax
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.freq_bins = fft_size // 2 + 1

        # Precompute band edges in frequency domain (Hz)
        self.band_edges = np.linspace(fmin, fmax, n_bands + 1)
        # Convert to bin indices
        self.band_bins = [
            (int(np.floor(f * fft_size / sample_rate)),
             int(np.ceil(f_hi * fft_size / sample_rate)))
            for f, f_hi in zip(self.band_edges[:-1], self.band_edges[1:])
        ]
        # Ensure bin indices are within valid range
        self.band_bins = [
            (max(0, lo), min(self.freq_bins - 1, hi))
            for lo, hi in self.band_bins
        ]

    def compute(self, spectrogram):
        """Compute spectral contrast features for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames), magnitude or power
            spectrum. Frequency bins should be linearly spaced.

        Returns
        -------
        dict
            Dictionary with keys 'contrast', 'peak', 'valley', each an array
            of shape (n_bands, time_frames).
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if spectrogram.shape[0] != self.freq_bins:
            raise ValueError(
                f"Spectrogram has {spectrogram.shape[0]} freq bins, expected {self.freq_bins}"
            )

        n_frames = spectrogram.shape[1]
        peak = np.zeros((self.n_bands, n_frames), dtype=np.float32)
        valley = np.zeros((self.n_bands, n_frames), dtype=np.float32)
        contrast = np.zeros((self.n_bands, n_frames), dtype=np.float32)

        for i, (lo, hi) in enumerate(self.band_bins):
            if hi <= lo:
                # Degenerate band, fill with zeros
                continue
            band = spectrogram[lo:hi+1, :]  # shape (band_bins, frames)
            if band.size == 0:
                continue
            # Peak = mean of the largest values (e.g., top 20%)
            # Valley = mean of the smallest values (e.g., bottom 20%)
            # Use percentiles for robustness
            sorted_band = np.sort(band, axis=0)
            n_bins = band.shape[0]
            n_peak = max(1, int(np.ceil(n_bins * 0.2)))
            n_valley = max(1, int(np.floor(n_bins * 0.2)))
            peak[i, :] = np.mean(sorted_band[-n_peak:, :], axis=0)
            valley[i, :] = np.mean(sorted_band[:n_valley, :], axis=0)
            contrast[i, :] = peak[i, :] - valley[i, :]

        return {
            'contrast': contrast,
            'peak': peak,
            'valley': valley,
        }

    def summary(self, spectrogram):
        """Compute a single summary vector per frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames).

        Returns
        -------
        np.ndarray
            Array of shape (n_bands, time_frames) containing the contrast.
        """
        return self.compute(spectrogram)['contrast']
