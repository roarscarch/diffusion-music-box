import numpy as np


class Spectrogram:
    """Convert between audio waveforms and spectrogram representations.

    This module provides a lightweight implementation of the short-time
    Fourier transform (STFT) and its inverse (ISTFT) for use in the
    diffusion music pipeline. Spectrogram tiles are 2D arrays where rows
    correspond to frequency bins and columns to time frames. The inverse
    transform reconstructs audio using overlap-add with a Hann window to
    minimize artifacts.

    Parameters
    ----------
    fft_size : int
        FFT size (number of frequency bins per frame). Must be a power of two.
    hop_length : int
        Number of samples between successive frames.
    sample_rate : int
        Sample rate of the audio.
    """

    def __init__(self, fft_size=1024, hop_length=256, sample_rate=22050):
        if fft_size & (fft_size - 1) != 0:
            raise ValueError("fft_size must be a power of two")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.freq_bins = fft_size // 2 + 1
        self.window = np.hanning(fft_size).astype(np.float32)
        # Normalize window for perfect reconstruction with overlap-add
        self.window /= np.sqrt(np.sum(self.window ** 2) / fft_size)

    def stft(self, audio):
        """Compute the magnitude spectrogram of an audio signal.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            2D float array of shape (freq_bins, n_frames) containing
            magnitude values.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be 1D")
        n_samples = len(audio)
        if n_samples < self.fft_size:
            raise ValueError("audio too short for FFT size")

        n_frames = 1 + (n_samples - self.fft_size) // self.hop_length
        spect = np.zeros((self.freq_bins, n_frames), dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.fft_size] * self.window
            spectrum = np.fft.rfft(frame)
            spect[:, i] = np.abs(spectrum)

        return spect

    def istft(self, spectrogram):
        """Reconstruct audio from a magnitude spectrogram.

        This method performs a simple inverse STFT using the phase from a
        previous iteration or a default. For real-time generation, the
        diffusion model typically operates on the magnitude and phase is
        derived separately. Here we assume zero phase for simplicity, which
        yields a synthetic but usable output.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D float array of shape (freq_bins, n_frames) containing
            magnitude values.

        Returns
        -------
        np.ndarray
            1D float array of reconstructed audio samples.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("spectrogram must be 2D")
        if spectrogram.shape[0] != self.freq_bins:
            raise ValueError(
                f"spectrogram has {spectrogram.shape[0]} freq bins, expected {self.freq_bins}"
            )
        n_frames = spectrogram.shape[1]
        n_samples = (n_frames - 1) * self.hop_length + self.fft_size
        audio = np.zeros(n_samples, dtype=np.float32)
        window_sum = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_length
            frame_spectrum = spectrogram[:, i]
            # Use zero phase (real part only) for reconstruction
            complex_spectrum = frame_spectrum.astype(np.complex64)
            frame = np.fft.irfft(complex_spectrum, n=self.fft_size)
            frame = frame * self.window
            audio[start:start + self.fft_size] += frame
            window_sum[start:start + self.fft_size] += self.window ** 2

        # Normalize by window overlap-add to avoid amplitude modulation
        nonzero = window_sum > 1e-8
        audio[nonzero] /= window_sum[nonzero]
        return audio

    def magnitude_to_audio(self, magnitude, phase=None):
        """Convert magnitude spectrogram to audio using provided or random phase.

        Parameters
        ----------
        magnitude : np.ndarray
            2D magnitude spectrogram.
        phase : np.ndarray, optional
            2D phase spectrogram (same shape). If None, random phase is used.

        Returns
        -------
        np.ndarray
            1D audio samples.
        """
        magnitude = np.asarray(magnitude, dtype=np.float32)
        if phase is None:
            rng = np.random.default_rng(0)
            phase = rng.uniform(-np.pi, np.pi, size=magnitude.shape).astype(np.float32)
        elif phase.shape != magnitude.shape:
            raise ValueError("phase must have same shape as magnitude")

        complex_spec = magnitude * np.exp(1j * phase)
        n_frames = complex_spec.shape[1]
        n_samples = (n_frames - 1) * self.hop_length + self.fft_size
        audio = np.zeros(n_samples, dtype=np.float32)
        window_sum = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = np.fft.irfft(complex_spec[:, i], n=self.fft_size)
            frame = frame * self.window
            audio[start:start + self.fft_size] += frame
            window_sum[start:start + self.fft_size] += self.window ** 2

        nonzero = window_sum > 1e-8
        audio[nonzero] /= window_sum[nonzero]
        return audio
