import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff point of a signal.

    The spectral rolloff is the frequency below which a specified percentage
    (typically 85% or 95%) of the total spectral energy is contained. It is
    a useful feature for characterizing the brightness of a sound and can be
    used to analyze or modulate generated ambient textures.
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_length=256, percentage=0.85):
        """
        Parameters
        ----------
        sample_rate : int
            Sample rate of the audio signal.
        fft_size : int
            FFT size (number of frequency bins).
        hop_length : int
            Hop length between frames in samples.
        percentage : float
            Fraction of total energy below the rolloff point (0 < p < 1).
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.percentage = percentage
        self.freq_bins = fft_size // 2 + 1

    def _stft(self, audio):
        """Compute short-time Fourier transform magnitudes.

        Returns a 2D array of shape (freq_bins, n_frames).
        """
        # Pad signal to ensure enough frames
        n_samples = len(audio)
        n_frames = 1 + (n_samples - self.fft_size) // self.hop_length
        if n_frames < 1:
            n_frames = 1
        # Pre-allocate spectrogram
        spectrogram = np.zeros((self.freq_bins, n_frames), dtype=np.float32)
        # Window function (Hann)
        window = np.hanning(self.fft_size).astype(np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.fft_size
            frame = audio[start:end]
            if len(frame) < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - len(frame)))
            spectrum = np.fft.rfft(frame * window, n=self.fft_size)
            spectrogram[:, i] = np.abs(spectrum)
        return spectrogram

    def compute(self, audio):
        """Compute spectral rolloff for each frame.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies (in Hz) for each frame.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Audio must be 1D")
        spectrogram = self._stft(audio)
        n_frames = spectrogram.shape[1]
        rolloff_freqs = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            spectrum = spectrogram[:, i]
            total_energy = np.sum(spectrum)
            if total_energy <= 0:
                rolloff_freqs[i] = 0.0
                continue
            cumsum = np.cumsum(spectrum)
            # Find the first bin where cumulative energy exceeds threshold
            threshold = self.percentage * total_energy
            bin_idx = np.searchsorted(cumsum, threshold, side='left')
            # Convert bin index to frequency
            freq = bin_idx * self.sample_rate / self.fft_size
            rolloff_freqs[i] = freq
        return rolloff_freqs

    def compute_average(self, audio):
        """Compute the mean spectral rolloff across frames.

        Useful as a single scalar feature for a segment.
        """
        rolloff = self.compute(audio)
        if len(rolloff) == 0:
            return 0.0
        return float(np.mean(rolloff))
