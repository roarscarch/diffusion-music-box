import numpy as np
from collections import OrderedDict


class TileCache:
    """A simple LRU cache for spectrogram tiles.

    This cache stores generated spectrogram tiles keyed by a string identifier.
    It uses an OrderedDict to implement least-recently-used eviction, ensuring
    that the most recently accessed tiles remain in memory. This is useful
    for reuse of tiles across generations, especially when the same tile is
    requested multiple times with the same parameters.

    Parameters
    ----------
    capacity : int
        Maximum number of tiles to hold in the cache.
    """

    def __init__(self, capacity=64):
        self.capacity = capacity
        self._cache = OrderedDict()

    def get(self, key):
        """Retrieve a tile from the cache by key.

        Parameters
        ----------
        key : str
            A unique identifier for the tile.

        Returns
        -------
        np.ndarray or None
            The cached tile as a 2D float array, or None if not found.
        """
        if key not in self._cache:
            return None
        # Move to end to mark as most recently used
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key, tile):
        """Store a tile in the cache.

        Parameters
        ----------
        key : str
            Unique identifier for the tile.
        tile : np.ndarray
            2D float array representing the spectrogram tile.

        Raises
        ------
        ValueError
            If the tile is not a 2D float array.
        """
        tile = np.asarray(tile, dtype=np.float32)
        if tile.ndim != 2:
            raise ValueError("Tile must be a 2D array")
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = tile
        if len(self._cache) > self.capacity:
            # Evict least recently used
            self._cache.popitem(last=False)

    def clear(self):
        """Remove all tiles from the cache."""
        self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key):
        return key in self._cache
