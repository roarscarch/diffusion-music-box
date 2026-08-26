import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of a spectrogram.

    The spectral rolloff is the frequency below which a given percentage of
    the total spectral energy is concentrated. It is a useful descriptor for
    characterizing the brightness or sharpness of a sound. This module
    provides a function to compute the rolloff for each time frame of a
    spectrogram.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal, used to convert frequency bins to Hz.
    rolloff_percent : float, optional
        Percentage (0.0 to 1.0) of total energy below which the rolloff is
        computed. Default is 0.85 (85%).
    """

    def __init__(self, sample_rate=22050, rolloff_percent=0.85):
        self.sample_rate = sample_rate
        self.rolloff_percent = rolloff_percent

    def compute(self, spectrogram):
        """Compute spectral rolloff for each frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) representing magnitude
            or power spectrogram.

        Returns
        -------
        np.ndarray
            1D array of length time_frames with rolloff frequencies in Hz.
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if not 0.0 < self.rolloff_percent <= 1.0:
            raise ValueError("rolloff_percent must be in (0, 1]")

        # Compute cumulative sum along frequency axis
        cumsum = np.cumsum(spectrogram, axis=0)
        total_energy = cumsum[-1, :]
        # Avoid division by zero for silent frames
        total_energy = np.maximum(total_energy, 1e-12)
        normalized = cumsum / total_energy

        # For each frame, find the first bin where cumulative energy exceeds threshold
        threshold = self.rolloff_percent
        # Create a mask of where normalized >= threshold
        mask = normalized >= threshold
        # Find first True index along axis 0 for each column
        rolloff_bins = np.argmax(mask, axis=0)

        # Convert bin index to frequency in Hz
        freq_resolution = self.sample_rate / (2 * (spectrogram.shape[0] - 1))
        rolloff_freqs = rolloff_bins * freq_resolution

        return rolloff_freqs.astype(np.float32)
