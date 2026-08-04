import numpy as np
from scipy import signal


class SpectrogramInverse:
    """Convert a spectrogram tile back to audio using overlap-add.

    The inverse transform uses a short-time Fourier transform (STFT) with
    a Hann window and 75% overlap. The magnitude spectrogram is combined
    with a random phase to produce a complex spectrogram, then inverted
    to a time-domain signal. This is suitable for generative audio where
    phase coherence is not critical.

    Parameters
    ----------
    n_fft : int
        FFT size. Must be even.
    hop_length : int
        Number of samples between successive frames.
    sample_rate : int
        Sample rate of the output audio.
    """

    def __init__(self, n_fft=512, hop_length=128, sample_rate=22050):
        if n_fft % 2 != 0:
            raise ValueError("n_fft must be even")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.window = signal.windows.hann(n_fft, sym=False)
        self._window_sum = None

    def invert(self, magnitude_tile):
        """Convert a magnitude spectrogram tile to audio.

        Parameters
        ----------
        magnitude_tile : np.ndarray
            2D array of shape (n_freq, n_frames). Values should be
            non-negative magnitudes (e.g., from a diffusion model).

        Returns
        -------
        np.ndarray
            1D float32 array of audio samples.
        """
        if magnitude_tile.ndim != 2:
            raise ValueError("magnitude_tile must be 2D")
        n_freq, n_frames = magnitude_tile.shape
        if n_freq != (self.n_fft // 2 + 1):
            raise ValueError(f"n_freq must equal {self.n_fft // 2 + 1}")

        # Generate random phase for each frame
        rng = np.random.default_rng()
        phase = rng.uniform(0, 2 * np.pi, size=(n_freq, n_frames))
        complex_spec = magnitude_tile * np.exp(1j * phase)

        # Reconstruct conjugate symmetric spectrogram
        full_spec = np.zeros((self.n_fft, n_frames), dtype=np.complex128)
        full_spec[:n_freq] = complex_spec
        # Mirror the positive frequencies (excluding Nyquist) to negative
        full_spec[n_freq:] = np.conj(complex_spec[-2:0:-1, :])

        # Invert STFT via overlap-add
        audio = np.zeros((n_frames - 1) * self.hop_length + self.n_fft, dtype=np.float64)
        window_sum = np.zeros_like(audio)

        for i in range(n_frames):
            frame = np.fft.irfft(full_spec[:, i], n=self.n_fft)
            start = i * self.hop_length
            end = start + self.n_fft
            audio[start:end] += frame * self.window
            window_sum[start:end] += self.window ** 2

        # Normalize by window sum to avoid amplitude modulation
        epsilon = 1e-8
        audio = audio / (window_sum + epsilon)
        # Trim trailing silence (last window is not fully covered)
        audio = audio[: (n_frames - 1) * self.hop_length + self.hop_length]
        return audio.astype(np.float32)

    def spectrogram_to_audio(self, magnitude_tile):
        """Alias for invert."""
        return self.invert(magnitude_tile)
