import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of an audio signal.

    The spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is contained. It is a measure
    of the spectral shape and can be used to distinguish between bright and
    dark sounds. This implementation computes the rolloff for a given frame
    of audio using the magnitude spectrum.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal.
    rolloff_percent : float, optional
        Percentage of total spectral energy to consider (default 0.85).
    """

    def __init__(self, sample_rate=22050, rolloff_percent=0.85):
        if not 0.0 < rolloff_percent <= 1.0:
            raise ValueError("rolloff_percent must be in (0, 1]")
        self.sample_rate = sample_rate
        self.rolloff_percent = rolloff_percent

    def compute(self, frames):
        """Compute spectral rolloff for each frame.

        Parameters
        ----------
        frames : np.ndarray
            Audio frames of shape (n_frames, frame_size) or (frame_size,).

        Returns
        -------
        np.ndarray
            Rolloff frequency in Hz for each frame. If input is 1D,
            returns a scalar.
        """
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim == 1:
            frames = frames[np.newaxis, :]
        if frames.ndim != 2:
            raise ValueError("frames must be 1D or 2D")

        n_frames, frame_size = frames.shape
        # Compute magnitude spectrum (one-sided)
        windowed = frames * np.hanning(frame_size)
        spectrum = np.fft.rfft(windowed, axis=1)
        magnitude = np.abs(spectrum)
        freqs = np.fft.rfftfreq(frame_size, 1.0 / self.sample_rate)

        # Compute cumulative energy
        energy = magnitude ** 2
        total_energy = np.sum(energy, axis=1)
        target_energy = total_energy * self.rolloff_percent

        # Find rolloff index for each frame
        cum_energy = np.cumsum(energy, axis=1)
        rolloff_indices = np.argmax(cum_energy >= target_energy[:, np.newaxis], axis=1)
        # If no index reaches threshold (shouldn't happen with positive energy)
        rolloff_indices = np.where(rolloff_indices == 0, 0, rolloff_indices)

        rolloff_freqs = freqs[rolloff_indices]
        return rolloff_freqs[0] if rolloff_freqs.size == 1 else rolloff_freqs

    def __call__(self, frames):
        """Callable interface for convenience."""
        return self.compute(frames)
