import numpy as np


class SpectrumInverse:
    """Inverse transform from spectrogram tiles to audio waveform.

    This module provides the inverse operation to the spectrogram analysis,
    converting a 2D frequency-time representation back into a 1D audio
    signal. It uses the inverse Short-Time Fourier Transform (ISTFT) with
    overlap-add synthesis to reconstruct the time-domain waveform. The class
    supports both magnitude-only spectrograms (using a random phase
    initialization) and complex spectrograms.

    Parameters
    ----------
    fft_size : int
        FFT size used during analysis.
    hop_length : int
        Hop length in samples between frames.
    window : str or np.ndarray, optional
        Window type to apply during synthesis. Can be a string ('hann',
        'hamming', 'blackman', etc.) or a precomputed window array. Defaults
        to 'hann'.
    """

    def __init__(self, fft_size=1024, hop_length=256, window='hann'):
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.freq_bins = fft_size // 2 + 1
        self.window = self._create_window(window)

    def _create_window(self, window):
        """Create a synthesis window array.

        Parameters
        ----------
        window : str or np.ndarray
            Either a window name or a precomputed array.

        Returns
        -------
        np.ndarray
            Window of length fft_size.
        """
        if isinstance(window, str):
            if window == 'hann':
                return np.hanning(self.fft_size)
            elif window == 'hamming':
                return np.hamming(self.fft_size)
            elif window == 'blackman':
                return np.blackman(self.fft_size)
            else:
                raise ValueError(f"Unknown window type: {window}")
        else:
            arr = np.asarray(window)
            if arr.ndim != 1 or len(arr) != self.fft_size:
                raise ValueError(f"Window must be 1D of length {self.fft_size}")
            return arr.astype(np.float32)

    def _istft(self, stft_matrix, phase=None):
        """Perform inverse STFT on a complex or magnitude spectrogram.

        Parameters
        ----------
        stft_matrix : np.ndarray
            Complex STFT matrix of shape (freq_bins, time_frames). If the
            matrix is real (magnitude), a random phase is used.
        phase : np.ndarray, optional
            Phase matrix of the same shape as stft_matrix. If not provided
            and stft_matrix is real, random phase is generated.

        Returns
        -------
        np.ndarray
            1D float audio signal.
        """
        if stft_matrix.ndim != 2:
            raise ValueError("STFT matrix must be 2D")
        if stft_matrix.shape[0] != self.freq_bins:
            raise ValueError(f"Expected {self.freq_bins} frequency bins, got {stft_matrix.shape[0]}")

        if np.iscomplexobj(stft_matrix):
            # Use provided complex matrix directly
            complex_stft = stft_matrix.astype(np.complex64)
        else:
            # Magnitude spectrogram: use given phase or random
            if phase is None:
                rng = np.random.default_rng()
                phase = rng.uniform(-np.pi, np.pi, size=stft_matrix.shape)
            if phase.shape != stft_matrix.shape:
                raise ValueError("Phase shape must match magnitude shape")
            complex_stft = stft_matrix * np.exp(1j * phase)

        n_frames = complex_stft.shape[1]
        expected_length = self.hop_length * (n_frames - 1) + self.fft_size
        output = np.zeros(expected_length, dtype=np.float32)
        window_sum = np.zeros(expected_length, dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.fft_size
            frame = np.fft.irfft(complex_stft[:, i], n=self.fft_size)
            frame = frame * self.window
            output[start:end] += frame
            window_sum[start:end] += self.window ** 2

        # Normalize by window sum to correct for overlap-add
        nonzero = window_sum > 1e-10
        output[nonzero] /= window_sum[nonzero]

        return output.astype(np.float32)

    def reconstruct(self, spectrogram, phase=None):
        """Reconstruct audio from a spectrogram tile.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames). Can be complex or
            magnitude-only.
        phase : np.ndarray, optional
            Phase matrix for magnitude-only spectrograms. If not given, a
            random phase is generated.

        Returns
        -------
        np.ndarray
            1D audio samples.
        """
        return self._istft(spectrogram, phase)
