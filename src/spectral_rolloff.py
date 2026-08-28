import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff frequency for a given magnitude spectrum.

    The spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is concentrated. It is a
    useful feature for characterizing the brightness and timbral shape of an
    audio signal.
    """

    def __init__(self, sample_rate=22050, rolloff_percent=0.85):
        """Initialize the spectral rolloff computer.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the audio signal in Hz.
        rolloff_percent : float
            Percentage of total spectral energy below the rolloff frequency.
            Must be between 0 and 1.
        """
        self.sample_rate = sample_rate
        self.rolloff_percent = rolloff_percent

    def compute(self, magnitude_spectrum):
        """Compute the spectral rolloff frequency for a magnitude spectrum.

        Parameters
        ----------
        magnitude_spectrum : np.ndarray
            1D array of magnitude values (e.g., from FFT).

        Returns
        -------
        float
            The spectral rolloff frequency in Hz.

        Raises
        ------
        ValueError
            If the magnitude spectrum is empty or rolloff_percent is out of range.
        """
        if magnitude_spectrum.ndim != 1:
            raise ValueError("Magnitude spectrum must be 1D")
        if magnitude_spectrum.size == 0:
            raise ValueError("Magnitude spectrum must not be empty")
        if not 0 < self.rolloff_percent < 1:
            raise ValueError("rolloff_percent must be between 0 and 1")

        cumulative = np.cumsum(magnitude_spectrum)
        total_energy = cumulative[-1]
        if total_energy == 0:
            return 0.0

        threshold = self.rolloff_percent * total_energy
        rolloff_index = np.searchsorted(cumulative, threshold)
        # Convert index to frequency
        fft_size = (magnitude_spectrum.size - 1) * 2
        freq_bin_width = self.sample_rate / fft_size
        return rolloff_index * freq_bin_width

    def compute_spectrogram(self, magnitude_spectrogram):
        """Compute spectral rolloff for each frame in a magnitude spectrogram.

        Parameters
        ----------
        magnitude_spectrogram : np.ndarray
            2D array of shape (n_frames, n_freq_bins) or (n_freq_bins, n_frames).
            The frame axis is determined by the larger dimension.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies for each frame.
        """
        magnitude_spectrogram = np.asarray(magnitude_spectrogram)
        if magnitude_spectrogram.ndim != 2:
            raise ValueError("Magnitude spectrogram must be 2D")

        # Assume shape is (n_frames, n_freq_bins) if rows > cols, else transpose
        if magnitude_spectrogram.shape[0] < magnitude_spectrogram.shape[1]:
            magnitude_spectrogram = magnitude_spectrogram.T

        rolloffs = np.zeros(magnitude_spectrogram.shape[0], dtype=np.float32)
        for i, frame in enumerate(magnitude_spectrogram):
            rolloffs[i] = self.compute(frame)
        return rolloffs
