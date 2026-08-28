import numpy as np


class SpectralEnvelope:
    """Extract the spectral envelope of a signal.

    The spectral envelope represents the overall shape of the spectrum,
    smoothing out fine spectral detail. It is useful for analyzing timbral
    characteristics and can be used to guide diffusion generation toward
    desired spectral contours.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal.
    n_fft : int, optional
        FFT size. Default is 2048.
    hop_length : int, optional
        Hop length between frames. Default is n_fft // 4.
    """

    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=None):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.freq_bins = n_fft // 2 + 1

    def _stft_magnitude(self, audio):
        """Compute the magnitude spectrogram using a simple STFT.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            2D float array of shape (freq_bins, n_frames) with magnitude values.
        """
        if audio.ndim != 1:
            raise ValueError("Audio must be 1D")
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        if n_frames <= 0:
            raise ValueError("Audio too short for given FFT size and hop length")
        window = np.hanning(self.n_fft)
        frames = np.zeros((self.freq_bins, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            segment = audio[start:start + self.n_fft] * window
            spectrum = np.fft.rfft(segment, n=self.n_fft)
            frames[:, i] = np.abs(spectrum)
        return frames

    def _smoothed_spectrum(self, magnitude, smoothing_bins=5):
        """Apply a simple moving average smoothing across frequency bins.

        Parameters
        ----------
        magnitude : np.ndarray
            2D magnitude spectrogram (freq_bins, n_frames).
        smoothing_bins : int, optional
            Number of bins to average over (must be odd). Default is 5.

        Returns
        -------
        np.ndarray
            Smoothed magnitude spectrogram.
        """
        if smoothing_bins % 2 == 0:
            raise ValueError("smoothing_bins must be odd")
        pad = smoothing_bins // 2
        # Pad along frequency axis
        padded = np.pad(magnitude, ((pad, pad), (0, 0)), mode='edge')
        # Simple moving average using cumulative sum for efficiency
        cumsum = np.cumsum(padded, axis=0)
        smoothed = np.zeros_like(magnitude, dtype=np.float32)
        for i in range(magnitude.shape[0]):
            start = i
            end = i + smoothing_bins
            # Since cumsum is over padded array, shift by pad
            total = cumsum[end] - cumsum[start]
            smoothed[i] = total / smoothing_bins
        return smoothed

    def extract(self, audio, smoothing_bins=5, normalize=True):
        """Extract the spectral envelope as a 2D array.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.
        smoothing_bins : int, optional
            Number of frequency bins to smooth over. Default is 5.
        normalize : bool, optional
            If True, normalize the envelope to [0, 1] per frame. Default is True.

        Returns
        -------
        np.ndarray
            2D float array of shape (freq_bins, n_frames) representing the
            spectral envelope magnitude.
        """
        magnitude = self._stft_magnitude(audio)
        envelope = self._smoothed_spectrum(magnitude, smoothing_bins)
        if normalize:
            # Normalize each frame to max 1.0
            max_vals = np.max(envelope, axis=0, keepdims=True)
            max_vals[max_vals == 0] = 1.0  # avoid division by zero
            envelope = envelope / max_vals
        return envelope

    def extract_frame(self, spectrum, smoothing_bins=5):
        """Extract the spectral envelope from a single magnitude spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            1D float array of magnitude spectrum (freq_bins,).
        smoothing_bins : int, optional
            Number of frequency bins to smooth over. Default is 5.

        Returns
        -------
        np.ndarray
            1D float array representing the spectral envelope.
        """
        if spectrum.ndim != 1:
            raise ValueError("Spectrum must be 1D")
        magnitude = spectrum.reshape(-1, 1)
        envelope = self._smoothed_spectrum(magnitude, smoothing_bins)
        return envelope.flatten()
