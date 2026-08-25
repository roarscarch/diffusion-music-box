import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of an audio signal.

    The spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is contained. It provides a
    measure of the brightness of the signal and is commonly used in audio
    analysis for timbre characterization.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz.
    fft_size : int, optional
        Size of the FFT window. Must be a power of two. Default is 1024.
    hop_length : int, optional
        Number of samples between successive frames. Default is 512.
    percentile : float, optional
        Percentage of total energy below the rolloff frequency. Must be
        between 0 and 1. Default is 0.85.
    """

    def __init__(self, sample_rate, fft_size=1024, hop_length=512, percentile=0.85):
        if fft_size <= 0 or (fft_size & (fft_size - 1)) != 0:
            raise ValueError("fft_size must be a power of two")
        if not 0 < percentile < 1:
            raise ValueError("percentile must be between 0 and 1")
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.percentile = percentile
        self._window = np.hanning(fft_size)

    def compute(self, audio):
        """Compute the spectral rolloff for each frame of the audio signal.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies in Hz for each frame.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be 1D")

        n_frames = max(1, (len(audio) - self.fft_size) // self.hop_length + 1)
        rolloff = np.zeros(n_frames, dtype=np.float32)

        bin_freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.fft_size]
            if len(frame) < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - len(frame)))
            spectrum = np.abs(np.fft.rfft(frame * self._window))
            total_energy = np.sum(spectrum ** 2)
            if total_energy == 0:
                rolloff[i] = 0.0
                continue
            cumulative = np.cumsum(spectrum ** 2)
            threshold = self.percentile * total_energy
            rolloff_idx = np.searchsorted(cumulative, threshold)
            rolloff[i] = bin_freqs[min(rolloff_idx, len(bin_freqs) - 1)]

        return rolloff
