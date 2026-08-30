import numpy as np


class SpectralBalance:
    """Analyze spectral balance features of an audio signal.

    This module computes metrics that describe the overall tonal balance
    of a spectrum, including the spectral centroid (brightness), spectral
    rolloff (where most energy is concentrated), and a balance ratio that
    compares low-frequency to high-frequency energy. These features are
    useful for understanding the timbral character of generated ambient
    music and for adjusting parameters to achieve a desired brightness.
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.freq_bins = fft_size // 2 + 1

    def _stft(self, audio):
        """Compute a simple STFT magnitude spectrogram.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        np.ndarray
            2D magnitude spectrogram of shape (freq_bins, frames).
        """
        # Pad signal to handle edge frames
        pad_len = self.fft_size // 2
        audio_padded = np.pad(audio, (pad_len, pad_len), mode='reflect')
        n_frames = 1 + (len(audio_padded) - self.fft_size) // self.hop_length
        frames = []
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio_padded[start:start + self.fft_size]
            window = np.hanning(self.fft_size)
            spec = np.fft.rfft(frame * window)
            frames.append(np.abs(spec))
        return np.array(frames).T  # shape (freq_bins, frames)

    def _freq_vector(self):
        """Return frequency values for each FFT bin."""
        return np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)

    def centroid(self, spectrogram=None, audio=None):
        """Compute spectral centroid (brightness) in Hz.

        The centroid is the weighted mean of frequencies, where weights are
        the magnitudes. Higher values indicate brighter, more high-frequency
        content.

        Parameters
        ----------
        spectrogram : np.ndarray, optional
            Magnitude spectrogram of shape (freq_bins, frames). If None,
            computed from audio.
        audio : np.ndarray, optional
            Audio samples used if spectrogram is None.

        Returns
        -------
        np.ndarray
            1D array of centroid values per frame.
        """
        if spectrogram is None:
            if audio is None:
                raise ValueError("Either spectrogram or audio must be provided")
            spectrogram = self._stft(audio)
        freqs = self._freq_vector()
        # Avoid division by zero by adding epsilon
        total = np.sum(spectrogram, axis=0) + 1e-10
        centroid = np.sum(spectrogram * freqs[:, np.newaxis], axis=0) / total
        return centroid

    def rolloff(self, spectrogram=None, audio=None, roll_percent=0.85):
        """Compute spectral rolloff frequency.

        The frequency below which `roll_percent` of the total energy is
        concentrated. Indicates where the spectral energy is concentrated.

        Parameters
        ----------
        spectrogram : np.ndarray, optional
            Magnitude spectrogram of shape (freq_bins, frames). If None,
            computed from audio.
        audio : np.ndarray, optional
            Audio samples used if spectrogram is None.
        roll_percent : float, optional
            Percentage of total energy (0.0 to 1.0) for the rolloff point.
            Default 0.85.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies per frame.
        """
        if spectrogram is None:
            if audio is None:
                raise ValueError("Either spectrogram or audio must be provided")
            spectrogram = self._stft(audio)
        freqs = self._freq_vector()
        cumsum = np.cumsum(spectrogram, axis=0)
        total = cumsum[-1, :] + 1e-10
        # Find the first bin where cumsum >= roll_percent * total
        threshold = roll_percent * total
        # For each frame, find index where cumsum crosses threshold
        # Use broadcasting to compare
        indices = np.argmax(cumsum >= threshold, axis=0)
        rolloff = freqs[indices]
        return rolloff

    def balance_ratio(self, spectrogram=None, audio=None, split_hz=1000):
        """Compute low-to-high frequency energy ratio.

        The ratio of energy below `split_hz` to energy above it. Values > 1
        indicate darker, bass-heavy content; values < 1 indicate brighter,
        treble-heavy content.

        Parameters
        ----------
        spectrogram : np.ndarray, optional
            Magnitude spectrogram of shape (freq_bins, frames). If None,
            computed from audio.
        audio : np.ndarray, optional
            Audio samples used if spectrogram is None.
        split_hz : float, optional
            Frequency (Hz) dividing low and high bands. Default 1000.

        Returns
        -------
        np.ndarray
            1D array of balance ratios per frame.
        """
        if spectrogram is None:
            if audio is None:
                raise ValueError("Either spectrogram or audio must be provided")
            spectrogram = self._stft(audio)
        freqs = self._freq_vector()
        low_mask = freqs <= split_hz
        low_energy = np.sum(spectrogram[low_mask, :], axis=0) + 1e-10
        high_energy = np.sum(spectrogram[~low_mask, :], axis=0) + 1e-10
        return low_energy / high_energy

    def compute_all(self, spectrogram=None, audio=None):
        """Compute all spectral balance features at once.

        Parameters
        ----------
        spectrogram : np.ndarray, optional
            Magnitude spectrogram of shape (freq_bins, frames). If None,
            computed from audio.
        audio : np.ndarray, optional
            Audio samples used if spectrogram is None.

        Returns
        -------
        dict
            Dictionary with keys 'centroid', 'rolloff', 'balance_ratio',
            each mapping to a 1D array of per-frame values.
        """
        if spectrogram is None:
            if audio is None:
                raise ValueError("Either spectrogram or audio must be provided")
            spectrogram = self._stft(audio)
        return {
            'centroid': self.centroid(spectrogram),
            'rolloff': self.rolloff(spectrogram),
            'balance_ratio': self.balance_ratio(spectrogram)
        }
