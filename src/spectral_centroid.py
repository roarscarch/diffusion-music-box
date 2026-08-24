import numpy as np


class SpectralCentroid:
    """Compute the spectral centroid of a spectrogram.

    The spectral centroid indicates where the "center of mass" of the spectrum
    is located. It is often used as a measure of brightness in audio signals.
    """

    @staticmethod
    def compute(spectrogram, sample_rate=22050, fft_size=1024):
        """Compute the spectral centroid for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitude
            spectrogram values.
        sample_rate : int, optional
            Sample rate in Hz (default 22050).
        fft_size : int, optional
            FFT size used to compute the spectrogram (default 1024).

        Returns
        -------
        np.ndarray
            1D array of length time_frames with the spectral centroid in Hz
            for each frame.
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        freq_bins, n_frames = spectrogram.shape
        # Frequency in Hz for each bin
        freqs = np.linspace(0, sample_rate / 2, freq_bins)
        # Compute weighted mean of frequencies
        with np.errstate(divide='ignore', invalid='ignore'):
            centroid = np.sum(freqs[:, None] * spectrogram, axis=0) / np.sum(spectrogram, axis=0)
        # Replace NaN (where sum is zero) with 0
        centroid = np.nan_to_num(centroid, nan=0.0)
        return centroid
