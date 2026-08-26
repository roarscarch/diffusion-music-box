import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of an audio signal.

    Spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is contained. It is a
    useful feature for distinguishing between bright and dark sounds
    and for analyzing the spectral shape of generated ambient audio.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal.
    fft_size : int, optional
        FFT size used for the spectrogram. Default is 1024.
    hop_length : int, optional
        Hop length in samples between frames. Default is 256.
    rolloff_percent : float, optional
        Percentage of total spectral energy to consider. Must be between
        0.0 and 1.0. Default is 0.85.
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_length=256,
                 rolloff_percent=0.85):
        if not 0.0 < rolloff_percent <= 1.0:
            raise ValueError("rolloff_percent must be in (0, 1]")
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.rolloff_percent = rolloff_percent
        self.freq_bins = fft_size // 2 + 1

    def compute(self, audio):
        """Compute spectral rolloff for each frame of the audio signal.

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

        # Compute STFT
        n_frames = 1 + (len(audio) - self.fft_size) // self.hop_length
        if n_frames <= 0:
            return np.array([], dtype=np.float32)

        rolloff = np.zeros(n_frames, dtype=np.float32)
        window = np.hanning(self.fft_size).astype(np.float32)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.fft_size]
            if len(frame) < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - len(frame)))
            spectrum = np.abs(np.fft.rfft(frame * window))
            power = spectrum ** 2
            total = np.sum(power)
            if total == 0:
                continue
            cumsum = np.cumsum(power)
            threshold = self.rolloff_percent * total
            # Find the first bin where cumulative energy exceeds threshold
            idx = np.searchsorted(cumsum, threshold)
            if idx >= len(cumsum):
                idx = len(cumsum) - 1
            rolloff[i] = idx * self.sample_rate / self.fft_size

        return rolloff

    def compute_from_spectrogram(self, spectrogram):
        """Compute spectral rolloff from a precomputed magnitude spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D float array of shape (freq_bins, n_frames) containing
            magnitude values.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies (in Hz) for each frame.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        n_frames = spectrogram.shape[1]
        rolloff = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            power = spectrogram[:, i] ** 2
            total = np.sum(power)
            if total == 0:
                continue
            cumsum = np.cumsum(power)
            threshold = self.rolloff_percent * total
            idx = np.searchsorted(cumsum, threshold)
            if idx >= len(cumsum):
                idx = len(cumsum) - 1
            rolloff[i] = idx * self.sample_rate / self.fft_size

        return rolloff
