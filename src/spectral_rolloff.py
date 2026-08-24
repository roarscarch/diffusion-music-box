import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff point for each frame of a spectrogram.

    The spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is contained. It is useful
    for characterising the brightness or sharpness of a sound and can be used
    for real-time timbre analysis in the diffusion music box.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio (used to map bin indices to frequencies).
    fft_size : int
        FFT size used to produce the spectrogram. The number of frequency bins
        is assumed to be fft_size // 2 + 1.
    percentile : float, optional
        The percentage of total energy below the rolloff point (0.0 to 1.0).
        Default is 0.85 (85%).
    """

    def __init__(self, sample_rate=22050, fft_size=1024, percentile=0.85):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.n_bins = fft_size // 2 + 1
        self.percentile = percentile
        if not 0.0 < self.percentile <= 1.0:
            raise ValueError("percentile must be in (0, 1]")

    def _bin_to_freq(self, bin_idx):
        """Convert a bin index to a frequency in Hz."""
        return bin_idx * self.sample_rate / self.fft_size

    def compute(self, spectrogram):
        """Compute the spectral rolloff for each frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (n_bins, n_frames) containing magnitude or
            power spectrogram values (non-negative).

        Returns
        -------
        np.ndarray
            1D array of length n_frames with the rolloff frequency in Hz for
            each frame.

        Raises
        ------
        ValueError
            If the spectrogram has the wrong shape or contains negative values.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if spectrogram.shape[0] != self.n_bins:
            raise ValueError(
                f"Expected {self.n_bins} frequency bins, got {spectrogram.shape[0]}"
            )
        if np.any(spectrogram < 0):
            raise ValueError("Spectrogram values must be non-negative")

        n_frames = spectrogram.shape[1]
        rolloff = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            frame = spectrogram[:, i]
            total_energy = np.sum(frame)
            if total_energy == 0:
                # No energy in the frame, set rolloff to 0 Hz
                rolloff[i] = 0.0
                continue

            cumulative = np.cumsum(frame)
            # Find the first bin where cumulative energy exceeds the threshold
            threshold = self.percentile * total_energy
            idx = np.searchsorted(cumulative, threshold)
            # Ensure idx is within bounds
            idx = min(idx, self.n_bins - 1)
            rolloff[i] = self._bin_to_freq(idx)

        return rolloff
