import numpy as np


class SpectrumInverse:
    """Perform inverse spectrogram transform to reconstruct audio from magnitude/phase.

    This module provides a real-time capable inverse transform that converts
    a 2D spectrogram (frequency-time) back to a 1D audio signal. It supports
    magnitude-only spectrograms using the Griffin-Lim algorithm, or complex
    spectrograms when phase information is available. The implementation
    uses overlap-add with a Hann window to ensure smooth reconstruction.

    Parameters
    ----------
    n_fft : int
        FFT size (number of frequency bins = n_fft // 2 + 1).
    hop_length : int
        Hop length in samples between time frames.
    window : str, optional
        Window function name ('hann', 'hamming', 'blackman'). Default 'hann'.
    """

    def __init__(self, n_fft=1024, hop_length=256, window='hann'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_name = window
        self.freq_bins = n_fft // 2 + 1
        self._window = self._get_window(window)

    def _get_window(self, name):
        """Return the window function array of length n_fft."""
        if name == 'hann':
            return np.hanning(self.n_fft)
        elif name == 'hamming':
            return np.hamming(self.n_fft)
        elif name == 'blackman':
            return np.blackman(self.n_fft)
        else:
            raise ValueError(f"Unknown window: {name}")

    def _istft(self, spectrogram, phase=None, iterations=32):
        """Inverse STFT using Griffin-Lim if phase is None.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitudes.
        phase : np.ndarray, optional
            2D array of same shape containing phase angles in radians.
            If None, Griffin-Lim algorithm is used.
        iterations : int
            Number of Griffin-Lim iterations. Ignored if phase is provided.

        Returns
        -------
        np.ndarray
            1D audio signal.
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if spectrogram.shape[0] != self.freq_bins:
            raise ValueError(f"Spectrogram frequency bins {spectrogram.shape[0]} != expected {self.freq_bins}")

        n_frames = spectrogram.shape[1]
        if phase is None:
            # Initialize phase randomly
            rng = np.random.default_rng(0)
            phase = rng.uniform(-np.pi, np.pi, size=spectrogram.shape)
            signal = self._griffin_lim(spectrogram, phase, iterations)
        else:
            if phase.shape != spectrogram.shape:
                raise ValueError("Phase must have same shape as spectrogram")
            signal = self._istft_with_phase(spectrogram, phase)
        return signal

    def _griffin_lim(self, magnitudes, initial_phase, iterations):
        """Griffin-Lim iterative phase reconstruction."""
        phase = initial_phase.copy()
        for _ in range(iterations):
            # Reconstruct signal with current phase
            complex_spec = magnitudes * np.exp(1j * phase)
            signal = self._istft_with_phase(complex_spec, None, use_complex=True)
            # Re-analyze to get new phase
            _, new_phase = self._stft(signal, return_phase=True)
            phase = new_phase
        # Final reconstruction
        complex_spec = magnitudes * np.exp(1j * phase)
        return self._istft_with_phase(complex_spec, None, use_complex=True)

    def _stft(self, signal, return_phase=False):
        """Forward STFT to analyze signal (used in Griffin-Lim).

        Returns magnitudes and optionally phases.
        """
        n_samples = len(signal)
        pad_len = self.n_fft - self.hop_length
        signal_padded = np.pad(signal, (pad_len // 2, pad_len // 2 + pad_len % 2), mode='reflect')
        n_frames = 1 + (len(signal_padded) - self.n_fft) // self.hop_length
        frames = np.zeros((n_frames, self.n_fft), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = signal_padded[start:start + self.n_fft] * self._window
        spec = np.fft.rfft(frames, axis=1)
        magnitudes = np.abs(spec)
        if return_phase:
            phases = np.angle(spec)
            return magnitudes.T, phases.T
        return magnitudes.T

    def _istft_with_phase(self, spectrogram, phase=None, use_complex=False):
        """Inverse STFT given magnitude and phase or complex spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            Magnitude (if phase given) or complex (if use_complex=True).
        phase : np.ndarray, optional
            Phase angles if spectrogram is magnitude.
        use_complex : bool
            If True, spectrogram is already complex.

        Returns
        -------
        np.ndarray
            Reconstructed audio signal.
        """
        if use_complex:
            complex_spec = spectrogram
        else:
            if phase is None:
                raise ValueError("Phase required when not use_complex")
            complex_spec = spectrogram * np.exp(1j * phase)

        # Inverse FFT
        frames = np.fft.irfft(complex_spec.T, n=self.n_fft, axis=1)
        # Apply window and overlap-add
        n_frames = frames.shape[0]
        output_len = (n_frames - 1) * self.hop_length + self.n_fft
        output = np.zeros(output_len, dtype=np.float32)
        window_sum = np.zeros(output_len, dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_length
            output[start:start + self.n_fft] += frames[i] * self._window
            window_sum[start:start + self.n_fft] += self._window ** 2

        # Normalize by window overlap
        nonzero = window_sum > 1e-10
        output[nonzero] /= window_sum[nonzero]

        # Trim to original length (assuming centered padding)
        pad_len = self.n_fft // 2
        if output_len > pad_len * 2:
            output = output[pad_len:-pad_len]
        else:
            output = output[pad_len:]
        return output

    def forward(self, spectrogram, phase=None, iterations=32):
        """Convert spectrogram to audio signal.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing magnitudes.
        phase : np.ndarray, optional
            2D array of phase angles. If None, Griffin-Lim is used.
        iterations : int, optional
            Number of Griffin-Lim iterations if phase is None.

        Returns
        -------
        np.ndarray
            1D audio signal.
        """
        return self._istft(spectrogram, phase, iterations)
