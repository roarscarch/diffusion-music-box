import numpy as np


def overlap_add(tiles, hop_length, window=None):
    """Overlap-add a sequence of spectrogram tiles into a single 2D array.

    Parameters
    ----------
    tiles : np.ndarray
        Array of shape (n_tiles, n_freq, tile_len) representing spectrogram tiles.
    hop_length : int
        Number of time steps between consecutive tiles (in samples along time axis).
    window : np.ndarray, optional
        Window function of length tile_len. If None, uses a Hann window.

    Returns
    -------
    np.ndarray
        Reconstructed 2D spectrogram of shape (n_freq, total_len).
    """
    tiles = np.asarray(tiles, dtype=np.float32)
    if tiles.ndim != 3:
        raise ValueError("tiles must be a 3D array (n_tiles, n_freq, tile_len)")
    n_tiles, n_freq, tile_len = tiles.shape
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if window is None:
        window = np.hanning(tile_len).astype(np.float32)
    else:
        window = np.asarray(window, dtype=np.float32)
        if window.ndim != 1 or window.shape[0] != tile_len:
            raise ValueError("window must be 1D of length tile_len")

    # Total length: start at 0, last tile starts at (n_tiles-1)*hop, ends at +tile_len
    total_len = (n_tiles - 1) * hop_length + tile_len
    out = np.zeros((n_freq, total_len), dtype=np.float32)
    norm = np.zeros(total_len, dtype=np.float32)

    for i in range(n_tiles):
        start = i * hop_length
        end = start + tile_len
        out[:, start:end] += tiles[i] * window[np.newaxis, :]
        norm[start:end] += window

    # Avoid division by zero
    norm = np.where(norm > 1e-8, norm, 1.0)
    out /= norm[np.newaxis, :]
    return out
