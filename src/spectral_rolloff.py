import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of an audio signal.

    The spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is contained. It provides a
    measure of the spectral shape, indicating where the majority of energy
    is concentrated. This can be useful for analyzing timbral brightness or
    for controlling diffusion parameters based on spectral characteristics.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz.
    fft_size : int
        FFT size to use for the spectrogram. Default is 1024.
    hop_length : int
        Hop length in samples between frames. Default is 512.
    threshold : float, optional
        Fraction of total energy below which the rolloff frequency is found.
        Must be between 0.0 and 1.0. Default is 0.85.
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_length=512, threshold=0.85):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self._window = np.hanning(fft_size)
        # Frequency bins for a one-sided spectrum
        self._freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    def _spectrogram(self, audio):
        """Compute the magnitude spectrogram of an audio signal.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            2D array of shape (n_frames, n_freq_bins) containing magnitude values.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be 1D")
        n_frames = 1 + (len(audio) - self.fft_size) // self.hop_length
        if n_frames <= 0:
            return np.zeros((0, len(self._freqs)), dtype=np.float32)
        # Pad the audio to ensure at least one frame
        pad_len = max(0, self.fft_size - len(audio))
        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len))
        frames = np.zeros((n_frames, self.fft_size), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = audio[start:start + self.fft_size] * self._window
        mag = np.abs(np.fft.rfft(frames, axis=1))
        return mag

    def compute(self, audio):
        """Compute the spectral rolloff for each frame.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies (in Hz) for each frame.
        """
        mag = self._spectrogram(audio)
        if mag.shape[0] == 0:
            return np.array([], dtype=np.float32)
        # Compute cumulative sum of energy
        cumsum = np.cumsum(mag ** 2, axis=1)
        total_energy = cumsum[:, -1:]
        # Avoid division by zero
        total_energy = np.where(total_energy == 0, 1.0, total_energy)
        # Find the first bin where cumulative energy exceeds threshold
        rolloff_idx = np.argmax(cumsum >= self.threshold * total_energy, axis=1)
        # If no bin exceeds (e.g., all zero), argmax returns 0; set to last bin
        # Handle the case where the threshold is never reached (all zeros)
        any_energy = np.any(mag > 0, axis=1)
        rolloff_idx = np.where(any_energy, rolloff_idx, len(self._freqs) - 1)
        return self._freqs[rolloff_idx]

    def compute_mean(self, audio):
        """Compute the mean spectral rolloff across all frames.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        float
            Mean rolloff frequency in Hz, or 0.0 if no frames.
        """
        rolloffs = self.compute(audio)
        if rolloffs.size == 0:
            return 0.0
        return float(np.mean(rolloffs))
