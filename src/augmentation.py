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

    def random_brightness(self, tile, brightness_range=(-0.1, 0.1)):
        """Add random brightness offset to the tile."""
        offset = self.rng.uniform(*brightness_range)
        return tile + offset

    def random_contrast(self, tile, contrast_range=(0.9, 1.1)):
        """Adjust contrast by scaling around the mean."""
        factor = self.rng.uniform(*contrast_range)
        mean = np.mean(tile)
        return (tile - mean) * factor + mean

    def random_frequency_mask(self, tile, num_masks=1, max_mask_width=4):
        """Apply random frequency masking to the tile (SpecAugment style).

        Masks a contiguous band of frequency bins by setting them to zero,
        which encourages the model to be robust to missing frequency info.

        Parameters
        ----------
        tile : np.ndarray
            2D spectrogram tile of shape (freq, time).
        num_masks : int
            Number of masks to apply.
        max_mask_width : int
            Maximum width of each mask in frequency bins.

        Returns
        -------
        np.ndarray
            Masked tile.
        """
        tile = np.array(tile, dtype=np.float32, copy=True)
        n_freq = tile.shape[-2]
        for _ in range(num_masks):
            width = int(self.rng.integers(1, max_mask_width + 1))
            start = int(self.rng.integers(0, n_freq - width + 1))
            tile[start:start+width, :] = 0.0
        return tile

    def augment(self, tile, time_shift=True, freq_shift=True, scale=True,
                brightness=True, contrast=True, frequency_mask=False):
        """Apply a random composition of augmentations to a tile.

        Parameters
        ----------
        tile : np.ndarray
            2D spectrogram tile of shape (freq, time).
        time_shift : bool, optional
            Whether to apply random time shift.
        freq_shift : bool, optional
            Whether to apply random frequency shift.
        scale : bool, optional
            Whether to apply random scaling.
        brightness : bool, optional
            Whether to apply random brightness.
        contrast : bool, optional
            Whether to apply random contrast.
        frequency_mask : bool, optional
            Whether to apply random frequency masking.

        Returns
        -------
        np.ndarray
            Augmented tile.
        """
        if time_shift:
            tile = self.random_time_shift(tile)
        if freq_shift:
            tile = self.random_freq_shift(tile)
        if scale:
            tile = self.random_scale(tile)
        if brightness:
            tile = self.random_brightness(tile)
        if contrast:
            tile = self.random_contrast(tile)
        if frequency_mask:
            tile = self.random_frequency_mask(tile)
        return tile
