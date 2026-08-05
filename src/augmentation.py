import numpy as np


class SpectrogramAugmenter:
    """Apply random augmentations to spectrogram tiles for data diversity.

    During generation, applying small random transforms to the noise or
    intermediate spectrogram tiles can produce more varied and evolving
    ambient textures. This module provides a set of lightweight,
    deterministic augmentations that operate on 2D spectrogram tiles.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def random_time_shift(self, tile, max_shift=8):
        """Shift the tile along the time axis by a random amount (circularly)."""
        if max_shift <= 0:
            return tile
        shift = self.rng.integers(-max_shift, max_shift + 1)
        if shift == 0:
            return tile
        return np.roll(tile, shift, axis=-1)

    def random_freq_shift(self, tile, max_shift=4):
        """Shift the tile along the frequency axis by a random amount (circularly)."""
        if max_shift <= 0:
            return tile
        shift = self.rng.integers(-max_shift, max_shift + 1)
        if shift == 0:
            return tile
        return np.roll(tile, shift, axis=-2)

    def random_scale(self, tile, scale_range=(0.9, 1.1)):
        """Scale the tile by a random factor."""
        scale = self.rng.uniform(*scale_range)
        return tile * scale

    def random_brightness(self, tile, delta_range=(-0.05, 0.05)):
        """Add a random constant offset to all values in the tile.

        This simulates a brightness change in the spectrogram, which can
        help the model adapt to varying overall intensity levels.

        Parameters
        ----------
        tile : np.ndarray
            Input spectrogram tile (2D array).
        delta_range : tuple of float, optional
            Range of additive offset values.

        Returns
        -------
        np.ndarray
            Brightness-augmented tile.
        """
        delta = self.rng.uniform(*delta_range)
        return tile + delta

    def random_contrast(self, tile, factor_range=(0.9, 1.1)):
        """Multiply the tile by a random contrast factor around the mean.

        This adjusts the dynamic range of the spectrogram, making it
        more or less contrasting relative to its mean value.

        Parameters
        ----------
        tile : np.ndarray
            Input spectrogram tile (2D array).
        factor_range : tuple of float, optional
            Range of contrast multiplication factors.

        Returns
        -------
        np.ndarray
            Contrast-augmented tile.
        """
        factor = self.rng.uniform(*factor_range)
        mean = tile.mean()
        return (tile - mean) * factor + mean
