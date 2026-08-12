import numpy as np


class SpectralContrast:
    """Compute spectral contrast features from a spectrogram.

    Spectral contrast describes the difference in amplitude between peaks
    and valleys in the spectrum, capturing the 'texture' of the sound.
    This module computes per-frame spectral contrast for a given spectrogram
    and can be used to modulate diffusion parameters or analyze generated
    audio.

    Parameters
    ----------
    n_bands : int, optional
        Number of octave bands to divide the spectrum into. Default is 6.
    fmin : float, optional
        Minimum frequency in Hz for the bands. Default is 200.0.
    fmax : float, optional
        Maximum frequency in Hz for the bands. Default is 8000.0.
    sample_rate : int, optional
        Sample rate of the audio. Default is 22050.
    """

    def __init__(self, n_bands=6, fmin=200.0, fmax=8000.0, sample_rate=22050):
        self.n_bands = n_bands
        self.fmin = fmin
        self.fmax = fmax
        self.sample_rate = sample_rate

    def _band_edges(self, n_fft):
        """Compute frequency bin edges for each band."""
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        # Ensure fmax doesn't exceed Nyquist
        fmax = min(self.fmax, self.sample_rate / 2)
        # Log-spaced band edges
        edges = np.geomspace(self.fmin, fmax, self.n_bands + 1)
        # Map to bin indices
        bin_edges = np.searchsorted(freqs, edges)
        # Ensure at least one bin per band
        bin_edges = np.clip(bin_edges, 1, len(freqs) - 1)
        return bin_edges

    def compute(self, spectrogram, n_fft=None):
        """Compute spectral contrast for each frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (n_freq_bins, n_frames) containing magnitudes.
        n_fft : int, optional
            FFT size used to generate the spectrogram. If None, inferred from
            spectrogram shape: n_fft = 2 * (n_freq_bins - 1).

        Returns
        -------
        np.ndarray
            2D array of shape (n_bands, n_frames) containing contrast values.
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        n_freq_bins, n_frames = spectrogram.shape
        if n_fft is None:
            n_fft = 2 * (n_freq_bins - 1)

        bin_edges = self._band_edges(n_fft)
        contrast = np.zeros((self.n_bands, n_frames), dtype=np.float32)

        for band in range(self.n_bands):
            start = bin_edges[band]
            end = bin_edges[band + 1]
            if end <= start:
                continue
            band_data = spectrogram[start:end, :]  # shape (n_bins_in_band, n_frames)
            if band_data.size == 0:
                continue
            # Peak = mean of top 10% values per frame
            # Valley = mean of bottom 10% values per frame
            sorted_data = np.sort(band_data, axis=0)
            n_vals = band_data.shape[0]
            n_peak = max(1, int(np.ceil(0.1 * n_vals)))
            n_valley = max(1, int(np.ceil(0.1 * n_vals)))
            peak = np.mean(sorted_data[-n_peak:, :], axis=0)
            valley = np.mean(sorted_data[:n_valley, :], axis=0)
            contrast[band, :] = peak - valley

        return contrast
