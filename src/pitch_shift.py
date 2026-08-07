import numpy as np


class PitchShifter:
    """Shift the pitch of a spectrogram tile by resampling along the frequency axis.

    This module provides a simple pitch-shifting effect for spectrogram tiles.
    Pitch shifting is achieved by resampling the frequency axis of the tile,
    which changes the perceived pitch while preserving the time evolution.
    The resampling is done via linear interpolation on the magnitude spectrum.

    Parameters
    ----------
    shift_semitones : float, optional
        Number of semitones to shift. Positive values raise pitch, negative lower.
    """

    def __init__(self, shift_semitones=0.0):
        self.shift_semitones = shift_semitones

    def shift(self, tile, shift_semitones=None):
        """Apply pitch shift to a spectrogram tile.

        Parameters
        ----------
        tile : np.ndarray
            2D spectrogram tile with shape (freq_bins, time_frames).
        shift_semitones : float, optional
            Override the shift amount. If None, use the value set at construction.

        Returns
        -------
        np.ndarray
            Pitch-shifted tile with same shape as input.
        """
        if shift_semitones is None:
            shift_semitones = self.shift_semitones
        if shift_semitones == 0:
            return tile.copy()

        # Convert semitones to a frequency scaling factor.
        # Each semitone corresponds to a multiplicative factor of 2^(1/12).
        scale_factor = 2 ** (shift_semitones / 12.0)

        freq_bins, time_frames = tile.shape
        # Original frequency bin centers (linearly spaced from 0 to 1).
        original_freqs = np.linspace(0.0, 1.0, freq_bins, endpoint=False)
        # New frequency positions after scaling.
        new_freqs = original_freqs * scale_factor
        # Only keep frequencies within the original range.
        valid = new_freqs < 1.0
        if not np.any(valid):
            # If all shifted frequencies are out of range, return a silent tile.
            return np.zeros_like(tile)

        # Build output tile by interpolating the magnitude spectrum.
        # We use linear interpolation along the frequency axis.
        output = np.zeros_like(tile)
        # For each output frequency bin, find the corresponding input position.
        # We map output bin index to an input index via inverse scaling.
        output_freqs = np.linspace(0.0, 1.0, freq_bins, endpoint=False)
        input_positions = output_freqs / scale_factor  # inverse of scaling
        # Clip positions to valid range [0, freq_bins-1]
        input_positions = np.clip(input_positions, 0, freq_bins - 1)
        # Linear interpolation
        left_indices = np.floor(input_positions).astype(int)
        right_indices = np.minimum(left_indices + 1, freq_bins - 1)
        frac = input_positions - left_indices
        # Interpolate across frequency for each time frame
        for t in range(time_frames):
            output[:, t] = (1 - frac) * tile[left_indices, t] + frac * tile[right_indices, t]

        # Apply a simple fade to avoid click at boundaries (optional)
        # Not needed for now.
        return output

    def __call__(self, tile, shift_semitones=None):
        """Callable interface for convenience."""
        return self.shift(tile, shift_semitones)
