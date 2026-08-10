import numpy as np


class SpectralGate:
    """Apply spectral gating to reduce noise in spectrogram tiles.

    Spectral gating is a common technique for noise reduction: for each
    frequency bin, we estimate a noise floor (e.g., from the lowest
    percentile of magnitudes over time) and then attenuate bins that fall
    below a threshold relative to that floor. This helps clean up the
    diffusion output, removing faint artifacts and improving the perceived
    audio quality.

    Parameters
    ----------
    threshold_db : float, optional
        Gain reduction in decibels applied to bins below the noise floor.
    floor_percentile : float, optional
        Percentile (0-100) of magnitude per frequency bin used to estimate
        the noise floor. Lower values give a more conservative estimate.
    attack_coefficient : float, optional
        Smoothing factor for the gate envelope (0-1). Higher values respond
        faster but may cause pumping.
    """

    def __init__(self, threshold_db=-40.0, floor_percentile=10.0, attack_coefficient=0.1):
        self.threshold_db = threshold_db
        self.floor_percentile = floor_percentile
        self.attack_coefficient = attack_coefficient

    def _noise_floor(self, magnitude):
        """Estimate noise floor per frequency bin.

        Parameters
        ----------
        magnitude : np.ndarray
            2D array of shape (freq_bins, time_steps) with magnitudes.

        Returns
        -------
        np.ndarray
            1D array of noise floor estimates per frequency bin.
        """
        # Use the specified percentile across time as the noise floor
        return np.percentile(magnitude, self.floor_percentile, axis=1)

    def apply(self, spectrogram):
        """Apply spectral gating to a spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D complex or real array of shape (freq_bins, time_steps).
            If complex, magnitude is used for gating and phase is preserved.

        Returns
        -------
        np.ndarray
            Gated spectrogram with same shape and dtype as input.
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")

        # Work with magnitude and phase if complex
        is_complex = np.iscomplexobj(spectrogram)
        if is_complex:
            magnitude = np.abs(spectrogram)
            phase = np.angle(spectrogram)
        else:
            magnitude = spectrogram

        # Estimate noise floor
        floor = self._noise_floor(magnitude)
        # Avoid division by zero
        floor = np.maximum(floor, 1e-12)

        # Compute gate gain in linear domain
        threshold_linear = 10.0 ** (self.threshold_db / 20.0)
        # For each bin, if magnitude is below floor * threshold, apply gain
        # We'll compute a smooth gain using a soft knee
        ratio = magnitude / floor[:, np.newaxis]
        # Simple hard gate: gain = 1 if above threshold, else threshold_linear
        gate = np.where(ratio > threshold_linear, 1.0, threshold_linear)

        # Apply attack smoothing across time to avoid clicks
        smoothed = np.empty_like(gate)
        smoothed[:, 0] = gate[:, 0]
        alpha = self.attack_coefficient
        for t in range(1, gate.shape[1]):
            # Attack fast, release slow? Here we use same coefficient for simplicity
            smoothed[:, t] = alpha * gate[:, t] + (1 - alpha) * smoothed[:, t - 1]

        # Apply gain
        gated_magnitude = magnitude * smoothed

        if is_complex:
            # Reconstruct complex spectrogram
            return gated_magnitude * np.exp(1j * phase)
        else:
            return gated_magnitude

    def __call__(self, spectrogram):
        """Convenience method to apply the gate."""
        return self.apply(spectrogram)
