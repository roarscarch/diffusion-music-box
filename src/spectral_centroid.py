import numpy as np


class SpectralCentroid:
    """Compute the spectral centroid of a spectrogram.

    The spectral centroid is a measure of the 'brightness' of a sound,
    defined as the weighted mean of the frequency bins, where the weights
    are the magnitudes of the corresponding bins. It is widely used in
    audio analysis to characterize timbral properties.

    This class provides both static and instance-based computation,
    supporting both single frames and full spectrograms.
    """

    def __init__(self, sample_rate=22050):
        """Initialize the spectral centroid calculator.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the audio signal. Used to convert bin indices
            to frequencies in Hertz.
        """
        self.sample_rate = sample_rate

    def compute(self, spectrogram, fft_size=None):
        """Compute the spectral centroid for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            Either a 1D array of magnitudes for a single frame
            (frequency bins) or a 2D array of shape (freq_bins, time_frames).
        fft_size : int, optional
            FFT size used to create the spectrogram. If None, the number of
            frequency bins is inferred from the spectrogram shape.

        Returns
        -------
        np.ndarray
            Array of spectral centroid values in Hertz. If the input is 1D,
            a scalar is returned; if 2D, an array of shape (time_frames,).
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim == 1:
            return self._compute_frame(spectrogram, fft_size)
        elif spectrogram.ndim == 2:
            freq_bins = spectrogram.shape[0]
            if fft_size is None:
                fft_size = (freq_bins - 1) * 2
            freqs = np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)
            if freqs.shape[0] != freq_bins:
                # If mismatch, recompute based on actual bins
                freqs = np.linspace(0, self.sample_rate / 2, freq_bins)
            # Weighted mean along frequency axis
            magnitudes = spectrogram
            total_mag = np.sum(magnitudes, axis=0)
            # Avoid division by zero
            total_mag[total_mag == 0] = 1e-12
            centroids = np.sum(freqs[:, np.newaxis] * magnitudes, axis=0) / total_mag
            return centroids
        else:
            raise ValueError("Spectrogram must be 1D or 2D")

    def _compute_frame(self, frame, fft_size=None):
        """Compute centroid for a single frame."""
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim != 1:
            raise ValueError("Frame must be 1D")
        freq_bins = len(frame)
        if fft_size is None:
            fft_size = (freq_bins - 1) * 2
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)
        if freqs.shape[0] != freq_bins:
            freqs = np.linspace(0, self.sample_rate / 2, freq_bins)
        total = np.sum(frame)
        if total == 0:
            return 0.0
        return np.sum(freqs * frame) / total

    def __call__(self, spectrogram, fft_size=None):
        """Callable interface."""
        return self.compute(spectrogram, fft_size)
