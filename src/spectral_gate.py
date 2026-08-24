import numpy as np


class SpectralGate:
    """Apply a noise gate to a spectrogram.

    This module suppresses low-amplitude frequency bins in a spectrogram,
    reducing background noise and emphasizing tonal content. It can be used
    as a preprocessing step before inverse transform to clean up generated
    audio segments.

    Parameters
    ----------
    threshold : float
        Relative amplitude threshold (0.0 to 1.0). Bins below this fraction
        of the maximum amplitude are attenuated.
    reduction_db : float
        Amount of attenuation in decibels applied to gated bins.
    """

    def __init__(self, threshold=0.1, reduction_db=20.0):
        self.threshold = threshold
        self.reduction_db = reduction_db

    def apply(self, spectrogram):
        """Apply the noise gate to a spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) or 3D with batch dimension.
            Magnitude spectrogram (non-negative).

        Returns
        -------
        np.ndarray
            Gated spectrogram of the same shape.
        """
        spec = np.asarray(spectrogram, dtype=np.float32)
        if spec.ndim == 2:
            return self._apply_2d(spec)
        elif spec.ndim == 3:
            # Apply per batch element
            return np.stack([self._apply_2d(s) for s in spec])
        else:
            raise ValueError("Spectrogram must be 2D or 3D")

    def _apply_2d(self, spec):
        """Apply gate to a single 2D spectrogram."""
        max_val = np.max(spec) if spec.size > 0 else 0.0
        if max_val <= 0:
            return spec

        gate_threshold = max_val * self.threshold
        reduction_factor = 10.0 ** (-self.reduction_db / 20.0)

        # Create a mask for bins below threshold
        mask = spec < gate_threshold

        # Apply reduction, keeping the original value if above threshold
        gated = np.where(mask, spec * reduction_factor, spec)
        return gated

    def __call__(self, spectrogram):
        """Alias for apply()."""
        return self.apply(spectrogram)
