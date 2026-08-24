import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of an audio signal.

    The spectral rolloff is the frequency below which a specified percentage
    (usually 85% or 95%) of the total spectral energy is contained. It is a
    useful feature for characterizing the brightness of a sound and can be
    used for timbre analysis or as a control parameter for generative music.
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_length=256, rolloff_percent=0.85):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.rolloff_percent = rolloff_percent
        self.freq_bins = fft_size // 2 + 1

    def _stft(self, audio):
        """Compute the magnitude spectrogram of the audio signal."""
        n_frames = 1 + (len(audio) - self.fft_size) // self.hop_length
        if n_frames < 1:
            n_frames = 1
        spectrogram = np.zeros((self.freq_bins, n_frames), dtype=np.float32)
        window = np.hanning(self.fft_size).astype(np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.fft_size]
            if len(frame) < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - len(frame)))
            spectrogram[:, i] = np.abs(np.fft.rfft(frame * window))
        return spectrogram

    def compute(self, audio):
        """Compute the spectral rolloff for each frame.

        Parameters
        ----------
        audio : np.ndarray
            1D audio samples.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies (in Hz) for each frame.
        """
        spectrogram = self._stft(audio)
        total_energy = np.sum(spectrogram ** 2, axis=0)
        cumulative = np.cumsum(spectrogram ** 2, axis=0)
        rolloff_bins = np.argmax(cumulative >= self.rolloff_percent * total_energy[None, :], axis=0)
        rolloff_bins = np.clip(rolloff_bins, 0, self.freq_bins - 1)
        rolloff_freqs = rolloff_bins * self.sample_rate / self.fft_size
        return rolloff_freqs
