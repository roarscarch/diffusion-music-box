import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of an audio signal.

    The spectral rolloff is the frequency below which a given percentage
    (typically 85%) of the total spectral energy is contained. It is a
    useful feature for distinguishing between harmonic and noise-like
    sounds, and can be used to analyze the brightness of the generated
    ambient music.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal.
    fft_size : int
        FFT size used for the spectrogram (number of frequency bins).
    hop_length : int
        Hop length in samples between time frames.
    rolloff_percent : float, optional
        Percentage of total energy below the rolloff frequency (0.0 to 1.0).
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_length=256,
                 rolloff_percent=0.85):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.rolloff_percent = rolloff_percent
        self.freq_bins = fft_size // 2 + 1

    def compute(self, audio):
        """Compute the spectral rolloff for each frame of audio.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies (in Hz) for each frame.
        """
        # Compute STFT magnitude spectrogram
        n_frames = 1 + (len(audio) - self.fft_size) // self.hop_length
        if n_frames <= 0:
            return np.array([], dtype=np.float32)

        # Pad audio to ensure full frames
        pad_len = self.fft_size + (n_frames - 1) * self.hop_length - len(audio)
        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len), mode='constant')

        # Use numpy's sliding_window_view to create frames
        frames = np.lib.stride_tricks.sliding_window_view(
            audio, self.fft_size)[::self.hop_length]

        # Apply Hann window to reduce spectral leakage
        window = np.hanning(self.fft_size).astype(np.float32)
        frames = frames * window

        # Compute magnitude spectrum
        spectrum = np.abs(np.fft.rfft(frames, axis=1))

        # Frequency bins in Hz
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)

        # Compute cumulative energy
        energy = spectrum ** 2
        total_energy = np.sum(energy, axis=1)
        # Avoid division by zero
        total_energy[total_energy == 0] = 1.0
        cumulative = np.cumsum(energy, axis=1) / total_energy[:, np.newaxis]

        # Find the first bin where cumulative energy exceeds the threshold
        rolloff_bins = np.argmax(cumulative >= self.rolloff_percent, axis=1)
        rolloff_freqs = freqs[rolloff_bins]

        return rolloff_freqs.astype(np.float32)
}
