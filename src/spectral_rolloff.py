import numpy as np


class SpectralRolloff:
    """Compute spectral rolloff of a spectrogram.

    Spectral rolloff is the frequency below which a specified percentage of
    the total spectral energy is contained. It is a measure of the spectral
    shape and can indicate brightness or timbral characteristics.
    """

    def __init__(self, threshold=0.85):
        """Initialize the spectral rolloff calculator.

        Parameters
        ----------
        threshold : float, optional
            Fraction of total energy below the rolloff frequency (default 0.85).
        """
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def compute(self, spectrogram, sample_rate, fft_size=None):
        """Compute spectral rolloff for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitude
            or power values.
        sample_rate : int
            Sample rate of the audio.
        fft_size : int, optional
            FFT size used to generate the spectrogram. If None, inferred
            from the number of frequency bins.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies in Hz for each time frame.
        """
        spec = np.asarray(spectrogram, dtype=np.float64)
        if spec.ndim != 2:
            raise ValueError("spectrogram must be 2D")
        if fft_size is None:
            fft_size = (spec.shape[0] - 1) * 2
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        if len(freqs) != spec.shape[0]:
            # Handle potential mismatch by trimming or padding
            if len(freqs) > spec.shape[0]:
                freqs = freqs[:spec.shape[0]]
            else:
                freqs = np.pad(freqs, (0, spec.shape[0] - len(freqs)), 'edge')

        cumulative = np.cumsum(spec, axis=0)
        total_energy = cumulative[-1, :]
        # Avoid division by zero
        total_energy = np.where(total_energy <= 0, 1e-12, total_energy)
        normalized = cumulative / total_energy

        # For each frame, find the first index where normalized >= threshold
        rolloff_indices = np.argmax(normalized >= self.threshold, axis=0)
        rolloff_freqs = freqs[rolloff_indices]

        return rolloff_freqs
