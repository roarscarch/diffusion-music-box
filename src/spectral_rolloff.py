import numpy as np


class SpectralRolloff:
    """Compute spectral rolloff frequency for each frame of a spectrogram.

    The spectral rolloff is the frequency below which a specified percentage
    of the total spectral energy is concentrated. It is a useful measure of
    the spectral shape and brightness of an audio signal. This module provides
    a function to compute the rolloff frequency for each time frame of a
    magnitude spectrogram or a power spectrogram.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz.
    rolloff_percent : float, optional
        Percentage of total spectral energy to consider (between 0 and 1).
        Default is 0.85 (85%).
    """

    def __init__(self, sample_rate=22050, rolloff_percent=0.85):
        if not 0.0 < rolloff_percent <= 1.0:
            raise ValueError("rolloff_percent must be in (0, 1]")
        self.sample_rate = sample_rate
        self.rolloff_percent = rolloff_percent

    def compute(self, spectrogram):
        """Compute the spectral rolloff frequency for each frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) representing the
            magnitude or power spectrogram. Frequencies are assumed to be
            linearly spaced from 0 to Nyquist.

        Returns
        -------
        np.ndarray
            1D array of length time_frames containing the rolloff frequency
            in Hz for each frame.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("spectrogram must be 2D")
        if spectrogram.shape[0] < 2:
            raise ValueError("spectrogram must have at least 2 frequency bins")

        freq_bins, time_frames = spectrogram.shape
        # Compute cumulative sum along frequency axis
        cumulative = np.cumsum(spectrogram, axis=0)
        total_energy = cumulative[-1, :]
        # Avoid division by zero; frames with zero energy get rolloff 0
        total_energy[total_energy == 0] = 1.0

        # Find the first bin where cumulative energy exceeds the threshold
        threshold = self.rolloff_percent * total_energy
        # For each frame, find the index where cumulative >= threshold
        # Use argmax on a boolean mask: first True index
        rolloff_bins = np.argmax(cumulative >= threshold, axis=0)

        # Convert bin index to frequency (Hz)
        # Frequency of bin i is i * sample_rate / (2 * (freq_bins - 1)) for
        # FFT with N bins (0 to Nyquist). But for simplicity, use linear mapping.
        max_freq = self.sample_rate / 2.0
        rolloff_freqs = rolloff_bins * max_freq / (freq_bins - 1)
        return rolloff_freqs

    def __call__(self, spectrogram):
        """Convenience call method."""
        return self.compute(spectrogram)
