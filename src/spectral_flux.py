import numpy as np


class SpectralFlux:
    """Compute spectral flux for detecting onsets and spectral changes.

    Spectral flux measures the frame-to-frame difference in magnitude spectra,
    making it useful for detecting note onsets, rhythmic events, and abrupt
    changes in the generated audio. This module provides a lightweight
    implementation that can be used for visualization, analysis, or to
    modulate diffusion parameters in real time.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio.
    fft_size : int
        FFT size used for the spectrogram (number of frequency bins).
    hop_length : int
        Hop length in samples between time frames.
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.freq_bins = fft_size // 2 + 1

    def flux(self, magnitude_frames):
        """Compute spectral flux across consecutive frames.

        Parameters
        ----------
        magnitude_frames : np.ndarray
            2D array of shape (n_frames, freq_bins) or (freq_bins, n_frames)
            containing magnitude spectra. The function detects orientation
            automatically based on the last dimension matching freq_bins.

        Returns
        -------
        np.ndarray
            1D array of spectral flux values, length n_frames - 1 (or 0 if
            fewer than 2 frames). The values are non-negative and represent
            the sum of positive differences between consecutive frames.
        """
        magnitude_frames = np.asarray(magnitude_frames, dtype=np.float32)
        if magnitude_frames.ndim != 2:
            raise ValueError("magnitude_frames must be a 2D array")

        # Determine orientation: expect (freq_bins, n_frames) or (n_frames, freq_bins)
        if magnitude_frames.shape[0] == self.freq_bins:
            # Already (freq_bins, n_frames)
            frames = magnitude_frames.T
        elif magnitude_frames.shape[1] == self.freq_bins:
            # (n_frames, freq_bins)
            frames = magnitude_frames
        else:
            # Fallback: assume (n_frames, freq_bins) and transpose if needed
            frames = magnitude_frames

        if frames.shape[0] < 2:
            return np.array([], dtype=np.float32)

        # Compute positive difference for each frame pair
        diff = np.diff(frames, axis=0)
        positive_diff = np.maximum(diff, 0)
        flux = np.sum(positive_diff, axis=1)
        return flux.astype(np.float32)

    def normalize(self, flux, eps=1e-8):
        """Normalize spectral flux values to [0, 1].

        Parameters
        ----------
        flux : np.ndarray
            Spectral flux values from :meth:`flux`.
        eps : float, optional
            Small value to avoid division by zero.

        Returns
        -------
        np.ndarray
            Normalized flux values in [0, 1] (or [0, 0] if all zeros).
        """
        flux = np.asarray(flux, dtype=np.float32)
        if flux.size == 0:
            return flux
        max_val = np.max(flux)
        if max_val < eps:
            return np.zeros_like(flux, dtype=np.float32)
        return (flux / max_val).astype(np.float32)

    def onset_strength(self, magnitude_frames, threshold=0.5):
        """Compute binary onset indicators based on normalized flux.

        Parameters
        ----------
        magnitude_frames : np.ndarray
            Same as :meth:`flux`.
        threshold : float, optional
            Normalized flux threshold above which a frame is considered an onset.

        Returns
        -------
        np.ndarray
            Boolean array of length n_frames - 1 indicating onsets.
        """
        flux = self.flux(magnitude_frames)
        if flux.size == 0:
            return np.array([], dtype=bool)
        norm = self.normalize(flux)
        return norm > threshold
