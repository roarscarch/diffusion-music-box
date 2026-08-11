import numpy as np


class CrossfadeMixer:
    """Blend audio segments with equal-power crossfading.

    This module provides a reusable crossfade algorithm for seamlessly
    transitioning between two audio segments. It supports both linear and
    equal-power (constant-power) crossfade curves, which are commonly used
    in audio applications to avoid clicks and volume dips during transitions.

    The mixer can be used in two ways:
    - As a one-shot function to blend two complete segments.
    - As an incremental processor that maintains an internal buffer for
      continuous streaming, where new segments are gradually blended into
      the output.
    """

    def __init__(self, crossfade_samples=256, curve='equal_power'):
        """Initialize the crossfade mixer.

        Parameters
        ----------
        crossfade_samples : int
            Number of samples over which to perform the crossfade.
        curve : str
            Type of crossfade curve: 'linear' or 'equal_power'.

        Raises
        ------
        ValueError
            If curve is not 'linear' or 'equal_power'.
        """
        if curve not in ('linear', 'equal_power'):
            raise ValueError(f"Unknown curve type: {curve}")
        self.crossfade_samples = max(1, int(crossfade_samples))
        self.curve = curve
        self._fade_in = self._compute_fade_in()
        self._fade_out = 1.0 - self._fade_in

    def _compute_fade_in(self):
        """Compute the fade-in envelope for the crossfade window."""
        n = self.crossfade_samples
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        if self.curve == 'linear':
            return t
        # Equal-power: sin/cos taper
        return np.sin(t * np.pi / 2.0) ** 2

    def crossfade(self, segment_a, segment_b):
        """Blend two segments with a crossfade at the boundary.

        Parameters
        ----------
        segment_a : np.ndarray
            First audio segment (1D float array).
        segment_b : np.ndarray
            Second audio segment (1D float array). Must have same length
            as segment_a.

        Returns
        -------
        np.ndarray
            Blended audio segment of the same length as inputs.

        Raises
        ------
        ValueError
            If segments have different lengths or crossfade is longer than
            the segments.
        """
        seg_a = np.asarray(segment_a, dtype=np.float32)
        seg_b = np.asarray(segment_b, dtype=np.float32)
        if seg_a.ndim != 1 or seg_b.ndim != 1:
            raise ValueError("Segments must be 1D arrays")
        if len(seg_a) != len(seg_b):
            raise ValueError("Segments must have the same length")
        if self.crossfade_samples > len(seg_a):
            raise ValueError("Crossfade length exceeds segment length")

        out = seg_a.copy()
        fade = self._fade_in
        # Apply crossfade over the last crossfade_samples of seg_a and first of seg_b
        start = len(seg_a) - self.crossfade_samples
        out[start:] = seg_a[start:] * self._fade_out + seg_b[:self.crossfade_samples] * fade
        return out

    def apply_overlay(self, base, overlay, start_index=0):
        """Overlay a segment onto a base signal with a crossfade entrance.

        This is useful for adding new material (e.g., arpeggios) on top of
        an existing ambient bed. The overlay is faded in over the first
        crossfade_samples and faded out at the end if it goes beyond the base.

        Parameters
        ----------
        base : np.ndarray
            Base audio signal (1D float array).
        overlay : np.ndarray
            Overlay signal to mix in (1D float array).
        start_index : int
            Sample index in the base where the overlay begins.

        Returns
        -------
        np.ndarray
            New array with the overlay mixed in. Length is max of base and
            overlay end.
        """
        base_arr = np.asarray(base, dtype=np.float32)
        overlay_arr = np.asarray(overlay, dtype=np.float32)
        if base_arr.ndim != 1 or overlay_arr.ndim != 1:
            raise ValueError("Signals must be 1D")

        end_index = start_index + len(overlay_arr)
        out_len = max(len(base_arr), end_index)
        out = np.zeros(out_len, dtype=np.float32)
        out[:len(base_arr)] = base_arr

        # Fade in the overlay
        fade_in_len = min(self.crossfade_samples, len(overlay_arr))
        fade_out_len = min(self.crossfade_samples, len(overlay_arr))

        # Create envelope for overlay
        env = np.ones(len(overlay_arr), dtype=np.float32)
        # Fade in
        if fade_in_len > 0:
            env[:fade_in_len] = self._fade_in[:fade_in_len]
        # Fade out at the end (if overlay extends beyond base, still fade out)
        if fade_out_len > 0:
            env[-fade_out_len:] *= self._fade_out[-fade_out_len:]

        out[start_index:end_index] += overlay_arr * env
        return out
