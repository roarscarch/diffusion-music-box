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
        """Add a random constant offset to the tile."""
        delta = self.rng.uniform(*delta_range)
        return tile + delta

    def random_noise(self, tile, noise_level=0.02):
        """Add small Gaussian noise to the tile."""
        noise = self.rng.normal(0, noise_level, tile.shape)
        return tile + noise

    def random_freq_mask(self, tile, max_masks=2, mask_width=4):
        """Randomly zero out a few frequency bands."""
        n_masks = self.rng.integers(0, max_masks + 1)
        for _ in range(n_masks):
            start = self.rng.integers(0, tile.shape[-2] - mask_width)
            tile[..., start:start+mask_width, :] = 0.0
        return tile

    def random_time_mask(self, tile, max_masks=2, mask_width=4):
        """Randomly zero out a few time steps."""
        n_masks = self.rng.integers(0, max_masks + 1)
        for _ in range(n_masks):
            start = self.rng.integers(0, tile.shape[-1] - mask_width)
            tile[..., :, start:start+mask_width] = 0.0
        return tile

    def apply_all(self, tile):
        """Apply a random combination of augmentations."""
        # Randomly select a subset of augmentation functions
        funcs = [
            self.random_time_shift,
            self.random_freq_shift,
            self.random_scale,
            self.random_brightness,
            self.random_noise,
            self.random_freq_mask,
            self.random_time_mask,
        ]
        # Shuffle and apply a random number (0 to all)
        self.rng.shuffle(funcs)
        n_apply = self.rng.integers(0, len(funcs) + 1)
        result = tile.copy()
        for fn in funcs[:n_apply]:
            result = fn(result)
        return result
