import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of an audio signal.

    Spectral rolloff is the frequency below which a specified percentage
    (typically 85%) of the total spectral energy is contained. It is a
    measure of the spectral shape and can be used to characterize the
    brightness or sharpness of a sound. This module provides a function to
    compute the rolloff for a given spectrogram (2D array of magnitude
    values) or for a time-domain signal using an STFT.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal.
    fft_size : int, optional
        FFT size used for the spectrogram. If None, the number of frequency
        bins is inferred from the input.
    hop_length : int, optional
        Hop length between frames (used only if computing from time-domain).
    percentile : float, optional
        Percentage of total energy below the rolloff frequency (0.0 to 1.0).
        Default is 0.85.
    """

    def __init__(self, sample_rate=22050, fft_size=None, hop_length=256, percentile=0.85):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.percentile = percentile
        if not 0.0 < self.percentile < 1.0:
            raise ValueError("percentile must be between 0.0 and 1.0")

    def _freq_bins(self, n_bins):
        """Return frequency values for each bin index."""
        return np.linspace(0, self.sample_rate / 2, n_bins, dtype=np.float32)

    def compute(self, spectrogram=None, magnitude=None):
        """Compute spectral rolloff for each frame.

        Parameters
        ----------
        spectrogram : np.ndarray, optional
            2D array of shape (n_freq_bins, n_frames) containing magnitude
            or power values. If None, magnitude must be provided.
        magnitude : np.ndarray, optional
            Alternative input: 2D array of magnitude values. Same as
            spectrogram. If both are None, raises ValueError.

        Returns
        -------
        np.ndarray
            1D array of length n_frames with rolloff frequencies in Hz.

        Raises
        ------
        ValueError
            If neither spectrogram nor magnitude is provided.
        """
        if spectrogram is not None:
            data = np.asarray(spectrogram, dtype=np.float32)
        elif magnitude is not None:
            data = np.asarray(magnitude, dtype=np.float32)
        else:
            raise ValueError("Either spectrogram or magnitude must be provided")

        if data.ndim != 2:
            raise ValueError("Input must be a 2D array")

        n_bins, n_frames = data.shape
        # Ensure non-negative values
        data = np.abs(data)

        # Compute cumulative energy along frequency axis
        cumsum = np.cumsum(data, axis=0)
        total_energy = cumsum[-1, :]
        # Avoid division by zero
        total_energy[total_energy == 0] = 1e-12
        normalized = cumsum / total_energy[np.newaxis, :]

        # Find the first bin where cumulative energy exceeds the threshold
        # For each frame, find the smallest index where normalized >= percentile
        # We'll use argmax on a boolean array, but that returns first True
        mask = normalized >= self.percentile
        # For frames with no True (shouldn't happen if percentile <=1 and total>0), fallback to last bin
        first_over = np.argmax(mask, axis=0)
        # If no True in a frame, argmax returns 0; but we can use np.where to set to n_bins-1
        has_over = np.any(mask, axis=0)
        first_over = np.where(has_over, first_over, n_bins - 1)

        freqs = self._freq_bins(n_bins)
        rolloff = freqs[first_over]
        return rolloff

    def compute_from_signal(self, signal):
        """Compute rolloff from a time-domain signal using STFT.

        Parameters
        ----------
        signal : np.ndarray
            1D array of audio samples.

        Returns
        -------
        np.ndarray
            Rolloff frequencies per frame.
        """
        if self.fft_size is None:
            raise ValueError("fft_size must be provided to compute from signal")
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D")

        # Compute STFT manually using numpy
        n_frames = 1 + (len(signal) - self.fft_size) // self.hop_length
        if n_frames <= 0:
            raise ValueError("Signal too short for given fft_size and hop_length")

        window = np.hanning(self.fft_size).astype(np.float32)
        spectrogram = np.zeros((self.fft_size // 2 + 1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = signal[start:start + self.fft_size] * window
            spectrum = np.fft.rfft(frame)
            spectrogram[:, i] = np.abs(spectrum)
        return self.compute(spectrogram)
