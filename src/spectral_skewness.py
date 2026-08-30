import numpy as np


class SpectralSkewness:
    """Compute spectral skewness of a spectrogram.

    Spectral skewness measures the asymmetry of the spectral distribution
    around its mean. Positive skewness indicates a longer tail towards
    higher frequencies, negative skewness towards lower frequencies.
    This feature is useful for analyzing timbral brightness and texture.

    Parameters
    ----------
    sample_rate : int, optional
        Sample rate of the audio (used for frequency axis, not required for
        skewness calculation but kept for API consistency).
    """

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def compute(self, spectrogram):
        """Compute spectral skewness for each time frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) or (time_frames, freq_bins).
            If shape[0] is time_frames, the input is transposed internally.
            Magnitude values are expected (non-negative).

        Returns
        -------
        np.ndarray
            1D array of shape (time_frames,) with skewness values per frame.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")

        # Determine orientation: if first dimension is larger, assume (freq_bins, time_frames)
        if spectrogram.shape[0] > spectrogram.shape[1]:
            # Already (freq_bins, time_frames)
            mag = spectrogram
        else:
            # Assume (time_frames, freq_bins) and transpose
            mag = spectrogram.T

        # Ensure non-negative magnitudes
        mag = np.abs(mag)

        # Frequency bin indices (0 to N-1) as the variable
        freq_idx = np.arange(mag.shape[0], dtype=np.float32)

        # Compute total energy per frame
        total = mag.sum(axis=0)
        total = np.maximum(total, 1e-10)  # avoid division by zero

        # Weighted mean frequency index
        mean = (freq_idx[:, None] * mag).sum(axis=0) / total

        # Standard deviation
        diff = freq_idx[:, None] - mean[None, :]
        variance = (mag * (diff ** 2)).sum(axis=0) / total
        std = np.sqrt(np.maximum(variance, 0))
        std = np.maximum(std, 1e-10)

        # Third central moment
        skew = (mag * (diff ** 3)).sum(axis=0) / total
        skew = skew / (std ** 3)

        return skew
