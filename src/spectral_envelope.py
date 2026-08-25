import numpy as np


class SpectralEnvelope:
    """Extract and apply the spectral envelope of a signal.

    The spectral envelope describes the overall shape of the magnitude
    spectrum, smoothing out fine-grained detail. It is useful for
    normalizing spectral content, shaping timbre, or analyzing the
    broad spectral characteristics of audio. This module provides methods
    to compute the envelope from a magnitude spectrogram and to apply it
    to a spectrogram for normalization or filtering.

    Parameters
    ----------
    fft_size : int
        FFT size used to produce the spectrogram (number of frequency bins
        is fft_size // 2 + 1).
    smoothing : float, optional
        Smoothing factor for the envelope (0.0 to 1.0). Higher values
        produce a smoother envelope by averaging over more neighboring
        frequency bins. Must be in the range [0, 1).
    """

    def __init__(self, fft_size=1024, smoothing=0.5):
        if fft_size < 2:
            raise ValueError("fft_size must be at least 2")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in range [0, 1)")
        self.fft_size = fft_size
        self.freq_bins = fft_size // 2 + 1
        self.smoothing = smoothing

    def compute_envelope(self, spectrogram):
        """Compute the spectral envelope of a magnitude spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) containing
            non-negative magnitude values.

        Returns
        -------
        np.ndarray
            2D array of the same shape as the input, containing the
            smoothed envelope. Each column is smoothed across frequency.

        Raises
        ------
        ValueError
            If the spectrogram does not have the expected number of
            frequency bins or contains negative values.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("spectrogram must be 2D")
        if spectrogram.shape[0] != self.freq_bins:
            raise ValueError(
                f"Expected {self.freq_bins} frequency bins, got {spectrogram.shape[0]}"
            )
        if np.any(spectrogram < 0):
            raise ValueError("spectrogram must contain non-negative values")

        # Smooth across frequency using a simple moving average
        kernel_size = max(1, int(self.smoothing * self.freq_bins))
        if kernel_size == 1:
            return spectrogram.copy()

        # Pad the spectrum symmetrically to handle edges
        pad = kernel_size // 2
        padded = np.pad(spectrogram, ((pad, pad), (0, 0)), mode='edge')
        envelope = np.zeros_like(spectrogram, dtype=np.float32)

        for i in range(self.freq_bins):
            envelope[i] = np.mean(padded[i:i + kernel_size], axis=0)

        return envelope

    def apply_envelope(self, spectrogram, envelope=None, normalize=True):
        """Apply a spectral envelope to a magnitude spectrogram.

        If an envelope is not provided, it is computed from the input
        spectrogram. The envelope is then applied by dividing the input
        by the envelope (if normalize is True) or by multiplying the
        envelope by the input (if normalize is False). This can be used
        to flatten the spectrum (normalize) or to shape the spectrum.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames).
        envelope : np.ndarray, optional
            Precomputed envelope of the same shape. If None, computed
            from the input.
        normalize : bool, optional
            If True, divide the spectrogram by the envelope to flatten
            the spectrum. If False, multiply the envelope by the
            spectrogram to shape it.

        Returns
        -------
        np.ndarray
            Processed spectrogram.

        Raises
        ------
        ValueError
            If the envelope shape does not match the spectrogram.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("spectrogram must be 2D")

        if envelope is None:
            envelope = self.compute_envelope(spectrogram)
        else:
            envelope = np.asarray(envelope, dtype=np.float32)
            if envelope.shape != spectrogram.shape:
                raise ValueError("envelope shape must match spectrogram shape")

        # Avoid division by zero
        safe_envelope = np.maximum(envelope, 1e-8)

        if normalize:
            return spectrogram / safe_envelope
        else:
            return spectrogram * envelope

    def __repr__(self):
        return (
            f"SpectralEnvelope(fft_size={self.fft_size}, "
            f"smoothing={self.smoothing})"
        )
