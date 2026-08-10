import numpy as np


class SpectrumInverse:
    """Inverse spectrogram transform to reconstruct audio from a magnitude spectrogram.

    This module provides functionality to convert a 2D spectrogram (frequency-time)
    back into a 1D audio signal. It uses the Griffin-Lim algorithm for phase
    recovery, which iteratively estimates the phase that best matches the given
    magnitude spectrogram. This is essential for playing back the generated
    spectrogram tiles as audible audio.

    Parameters
    ----------
    fft_size : int
        FFT size used for the spectrogram (must be even).
    hop_length : int
        Hop length in samples between time frames.
    sample_rate : int
        Sample rate of the audio (used for potential resampling).
    """

    def __init__(self, fft_size=1024, hop_length=256, sample_rate=22050):
        if fft_size % 2 != 0:
            raise ValueError("fft_size must be even")
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.freq_bins = fft_size // 2 + 1

    def _stft(self, audio):
        """Compute short-time Fourier transform of audio.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            Complex STFT of shape (freq_bins, n_frames).
        """
        n_frames = 1 + (len(audio) - self.fft_size) // self.hop_length
        if n_frames <= 0:
            return np.zeros((self.freq_bins, 1), dtype=np.complex64)
        stft = np.zeros((self.freq_bins, n_frames), dtype=np.complex64)
        window = np.hanning(self.fft_size).astype(np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.fft_size] * window
            stft[:, i] = np.fft.rfft(frame)
        return stft

    def _istft(self, stft, length=None):
        """Compute inverse STFT from complex spectrogram.

        Parameters
        ----------
        stft : np.ndarray
            Complex STFT of shape (freq_bins, n_frames).
        length : int, optional
            Desired output length. If None, inferred from frames.

        Returns
        -------
        np.ndarray
            1D float array of reconstructed audio.
        """
        n_frames = stft.shape[1]
        if length is None:
            length = (n_frames - 1) * self.hop_length + self.fft_size
        audio = np.zeros(length, dtype=np.float32)
        window = np.hanning(self.fft_size).astype(np.float32)
        window_sum = np.zeros(length, dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = np.fft.irfft(stft[:, i], n=self.fft_size)
            end = min(start + self.fft_size, length)
            actual_len = end - start
            if actual_len < self.fft_size:
                frame = frame[:actual_len]
                win = window[:actual_len]
            else:
                win = window
            audio[start:end] += frame * win
            window_sum[start:end] += win * win
        # Normalize by window overlap-sum to avoid amplitude modulation
        nonzero = window_sum > 1e-8
        audio[nonzero] /= window_sum[nonzero]
        return audio

    def griffin_lim(self, magnitude, iterations=30, phase=None, length=None):
        """Reconstruct audio from a magnitude spectrogram using Griffin-Lim.

        Parameters
        ----------
        magnitude : np.ndarray
            2D float array of shape (freq_bins, n_frames) with non-negative values.
        iterations : int, optional
            Number of iterations for phase recovery.
        phase : np.ndarray, optional
            Initial complex phase estimate of shape (freq_bins, n_frames).
            If None, starts with random phase.
        length : int, optional
            Desired output length. If None, inferred from frames.

        Returns
        -------
        np.ndarray
            1D float array of reconstructed audio.
        """
        if magnitude.shape[0] != self.freq_bins:
            raise ValueError(f"Expected {self.freq_bins} frequency bins, got {magnitude.shape[0]}")
        n_frames = magnitude.shape[1]
        if length is None:
            length = (n_frames - 1) * self.hop_length + self.fft_size

        # Initialize phase
        if phase is None:
            rng = np.random.default_rng(0)
            phase = np.exp(2j * np.pi * rng.random(magnitude.shape)).astype(np.complex64)
        else:
            if phase.shape != magnitude.shape:
                raise ValueError("phase shape must match magnitude shape")
            phase = phase.astype(np.complex64)

        # Iterative phase recovery
        for _ in range(iterations):
            # Combine magnitude with current phase
            stft = magnitude * phase
            # Inverse STFT to time domain
            audio = self._istft(stft, length=length)
            # Forward STFT to get new phase
            new_stft = self._stft(audio)
            # Update phase to match new STFT's phase, keep magnitude
            phase = np.exp(1j * np.angle(new_stft)).astype(np.complex64)

        # Final reconstruction
        stft = magnitude * phase
        audio = self._istft(stft, length=length)
        return audio

    def reconstruct(self, magnitude, iterations=30, phase=None, length=None):
        """Alias for griffin_lim for convenience."""
        return self.griffin_lim(magnitude, iterations=iterations, phase=phase, length=length)
