import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of an audio signal.

    The spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is contained. It is a useful
    feature for distinguishing between different types of sounds, such as
    voiced vs unvoiced speech, and can be used to characterize the brightness
    of an audio signal.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz.
    percentile : float, optional
        The percentage of total spectral energy below which the rolloff is
        computed. Must be between 0 and 1. Default is 0.85.
    """

    def __init__(self, sample_rate=22050, percentile=0.85):
        self.sample_rate = sample_rate
        self.percentile = percentile
        if not 0.0 < self.percentile < 1.0:
            raise ValueError("Percentile must be between 0 and 1")

    def compute(self, audio, frame_size=1024, hop_length=256):
        """Compute the spectral rolloff for each frame of an audio signal.

        Parameters
        ----------
        audio : np.ndarray
            Input audio signal as a 1D float array.
        frame_size : int, optional
            Size of each analysis frame in samples. Default is 1024.
        hop_length : int, optional
            Number of samples between consecutive frames. Default is 256.

        Returns
        -------
        np.ndarray
            1D array of spectral rolloff frequencies (in Hz) for each frame.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Audio must be 1D")

        # Zero-pad the audio so that we can take complete frames
        num_frames = 1 + (len(audio) - frame_size) // hop_length
        if num_frames <= 0:
            return np.array([], dtype=np.float32)

        rolloffs = np.zeros(num_frames, dtype=np.float32)
        window = np.hanning(frame_size)
        # Precompute frequency values for each bin
        freqs = np.fft.rfftfreq(frame_size, d=1.0 / self.sample_rate)

        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + frame_size] * window
            spectrum = np.abs(np.fft.rfft(frame))
            total_energy = np.sum(spectrum)
            if total_energy == 0:
                rolloffs[i] = 0.0
                continue
            cumulative = np.cumsum(spectrum)
            # Find the index where cumulative energy reaches the threshold
            threshold = self.percentile * total_energy
            idx = np.searchsorted(cumulative, threshold)
            # Ensure idx is within bounds
            idx = min(idx, len(freqs) - 1)
            rolloffs[i] = freqs[idx]

        return rolloffs

    def __call__(self, audio, frame_size=1024, hop_length=256):
        """Callable interface for compute()."""
        return self.compute(audio, frame_size, hop_length)
