import numpy as np


class SpectrogramNormalizer:
    """Normalize spectrogram tiles to a consistent scale.

    This module provides functions to normalize spectrogram tiles so that
    their magnitude values fall within a predictable range. This is useful
    for feeding tiles into a diffusion model, where consistent input scaling
    improves stability and convergence. The normalizer supports two modes:
    min-max scaling and z-score standardization.

    Parameters
    ----------
    mode : str, optional
        Normalization mode: 'minmax' or 'zscore'. Default is 'minmax'.
    eps : float, optional
        Small epsilon to avoid division by zero.
    """

    def __init__(self, mode='minmax', eps=1e-8):
        if mode not in ('minmax', 'zscore'):
            raise ValueError("mode must be 'minmax' or 'zscore'")
        self.mode = mode
        self.eps = eps

    def normalize(self, tile):
        """Normalize a spectrogram tile.

        Parameters
        ----------
        tile : np.ndarray
            2D spectrogram tile (frequencies x time).

        Returns
        -------
        np.ndarray
            Normalized tile with same shape as input.
        """
        tile = np.asarray(tile, dtype=np.float32)
        if tile.ndim != 2:
            raise ValueError("tile must be 2D")
        if self.mode == 'minmax':
            return self._minmax(tile)
        else:
            return self._zscore(tile)

    def denormalize(self, tile, original_stats=None):
        """Denormalize a tile back to the original scale.

        Parameters
        ----------
        tile : np.ndarray
            Normalized tile (frequencies x time).
        original_stats : tuple, optional
            Tuple of (min, max) for minmax or (mean, std) for zscore.
            If None, uses the tile's own statistics (only appropriate if the
            tile was normalized independently).

        Returns
        -------
        np.ndarray
            Denormalized tile.
        """
        tile = np.asarray(tile, dtype=np.float32)
        if tile.ndim != 2:
            raise ValueError("tile must be 2D")
        if self.mode == 'minmax':
            if original_stats is None:
                raise ValueError("original_stats required for minmax denormalization")
            min_val, max_val = original_stats
            return tile * (max_val - min_val) + min_val
        else:
            if original_stats is None:
                raise ValueError("original_stats required for zscore denormalization")
            mean, std = original_stats
            return tile * std + mean

    def _minmax(self, tile):
        min_val = np.min(tile)
        max_val = np.max(tile)
        if max_val - min_val < self.eps:
            return np.zeros_like(tile)
        return (tile - min_val) / (max_val - min_val)

    def _zscore(self, tile):
        mean = np.mean(tile)
        std = np.std(tile)
        if std < self.eps:
            return np.zeros_like(tile)
        return (tile - mean) / std

    def get_stats(self, tile):
        """Return the statistics needed to denormalize a tile.

        Parameters
        ----------
        tile : np.ndarray
            Original tile before normalization.

        Returns
        -------
        tuple
            For minmax: (min, max). For zscore: (mean, std).
        """
        tile = np.asarray(tile, dtype=np.float32)
        if self.mode == 'minmax':
            return (np.min(tile), np.max(tile))
        else:
            return (np.mean(tile), np.std(tile))
