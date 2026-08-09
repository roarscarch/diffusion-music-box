import numpy as np
import threading
from collections import OrderedDict


class TileCache:
    """A thread-safe LRU cache for spectrogram tiles.

    The diffusion process generates tiles that are expensive to compute.
    This cache stores recently generated tiles keyed by a hash of the
    input parameters (noise seed, diffusion steps, etc.) to avoid
    recomputing identical tiles when parameters haven't changed.
    """

    def __init__(self, capacity=64):
        self.capacity = max(1, capacity)
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def _make_key(self, *args, **kwargs):
        """Create a hashable key from arguments."""
        key_parts = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append((arg.shape, arg.tobytes()))
            else:
                key_parts.append(arg)
        for k, v in sorted(kwargs.items()):
            if isinstance(v, np.ndarray):
                key_parts.append((k, v.shape, v.tobytes()))
            else:
                key_parts.append((k, v))
        try:
            return hash(tuple(key_parts))
        except TypeError:
            return hash(str(key_parts))

    def get(self, *args, **kwargs):
        """Retrieve a cached tile if it exists.

        Returns None if the tile is not in the cache.
        """
        key = self._make_key(*args, **kwargs)
        with self._lock:
            if key in self._cache:
                # Move to end to mark as recently used
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, tile, *args, **kwargs):
        """Store a tile in the cache.

        Parameters
        ----------
        tile : np.ndarray
            The spectrogram tile to cache.
        *args, **kwargs
            Parameters that uniquely identify the tile.
        """
        key = self._make_key(*args, **kwargs)
        with self._lock:
            self._cache[key] = tile
            self._cache.move_to_end(key)
            if len(self._cache) > self.capacity:
                # Remove the oldest item (least recently used)
                self._cache.popitem(last=False)

    def clear(self):
        """Clear all cached tiles."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self):
        """Current number of cached tiles."""
        with self._lock:
            return len(self._cache)

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"TileCache(capacity={self.capacity}, size={self.size})"
