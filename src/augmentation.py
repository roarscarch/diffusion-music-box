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
        factor = self.rng.uniform(*scale_range)
        return tile * factor

    def random_noise(self, tile, noise_level=0.01):
        """Add a small amount of Gaussian noise to the tile."""
        noise = self.rng.normal(0, noise_level, tile.shape)
        return tile + noise

    def random_mix(self, tile1, tile2, mix_range=(0.0, 0.2)):
        """Mix a small amount of another tile into this one."""
        alpha = self.rng.uniform(*mix_range)
        return (1 - alpha) * tile1 + alpha * tile2

    def apply_random(self, tile, transforms=None):
        """Apply a random subset of augmentations.

        Parameters
        ----------
        tile : np.ndarray
            Input spectrogram tile of shape (n_freq, n_frames).
        transforms : list of str, optional
            List of augmentation names to consider. If None, all are used.

        Returns
        -------
        np.ndarray
            Augmented tile.
        """
        if transforms is None:
            transforms = ['time_shift', 'freq_shift', 'scale', 'noise']

        # Shuffle and apply a random subset (at least one)
        n = self.rng.integers(1, len(transforms) + 1)
        chosen = self.rng.choice(transforms, size=n, replace=False)

        result = np.array(tile, dtype=float)
        for name in chosen:
            if name == 'time_shift':
                result = self.random_time_shift(result)
            elif name == 'freq_shift':
                result = self.random_freq_shift(result)
            elif name == 'scale':
                result = self.random_scale(result)
            elif name == 'noise':
                result = self.random_noise(result)
        return result
