import numpy as np
import threading


class TileCache:
    """A thread-safe cache for generated spectrogram tiles.

    This cache stores 2D spectrogram tiles keyed by a tuple of parameters
    (e.g., seed, noise schedule, diffusion steps). It prevents redundant
    generation of identical tiles and speeds up real-time playback by
    reusing previously computed tiles.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of tiles to keep in the cache. When full, the oldest
        entries are evicted (FIFO).
    """

    def __init__(self, max_size=128):
        self._cache = {}