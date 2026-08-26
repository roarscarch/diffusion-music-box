import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of a spectrogram.

    The spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is contained. It is a useful
    feature for characterizing the brightness of a sound and can be used to
    modulate the diffusion parameters in real time.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal.
    rolloff_percent : float, optional
        Percentage of total energy below the rolloff frequency (0.0 to 1.0).
        Default is 0.85.
    """

    def __init__(self, sample_rate=22050, rolloff_percent=0.85):
        self.sample_rate = sample_rate
        self.rolloff_percent = rolloff_percent

    def compute(self, spectrogram):
        """Compute the spectral rolloff for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitude
            spectrogram values (real, non-negative).

        Returns
        -------
        np.ndarray
            1D array of length time_frames with the rolloff frequency in Hz
            for each frame.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")

        freq_bins, n_frames = spectrogram.shape
        # Compute cumulative sum along frequency axis
        cumulative = np.cumsum(spectrogram, axis=0)
        total_energy = cumulative[-1, :]

        # Avoid division by zero for silent frames
        total_energy = np.where(total_energy == 0, 1.0, total_energy)
        normalized_cumulative = cumulative / total_energy

        # Find the first bin where cumulative energy exceeds the threshold
        # For each frame, get the index of the first bin >= rolloff_percent
        # Use argmax on a boolean array (first True index)
        threshold = self.rolloff_percent
        rolloff_bins = np.argmax(normalized_cumulative >= threshold, axis=0)

        # Convert bin index to frequency (Hz)
        # Frequency of bin i is i * sample_rate / fft_size, but we don't have fft_size.
        # Instead, we assume bin spacing is sample_rate / (2 * (freq_bins - 1)).
        # This is a reasonable approximation for a one-sided spectrum.
        bin_width = self.sample_rate / (2.0 * (freq_bins - 1))
        rolloff_freq = rolloff_bins * bin_width

        return rolloff_freq
