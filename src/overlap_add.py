import numpy as np


def crossfade(segments, fade_samples=256, window='hann'):
    """Stitch multiple audio segments together with crossfade.

    Parameters
    ----------
    segments : list of np.ndarray
        List of 1D float arrays (audio segments).
    fade_samples : int
        Number of samples for the crossfade overlap.
    window : str
        Window type for the crossfade ('hann' or 'linear').

    Returns
    -------
    np.ndarray
        The concatenated audio with crossfades applied.
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)

    # Convert to float32 and ensure 1D
    segs = [np.asarray(s, dtype=np.float32).ravel() for s in segments]

    # If only one segment, return it as is
    if len(segs) == 1:
        return segs[0]

    if fade_samples < 0:
        raise ValueError("fade_samples must be non-negative")

    # Create window
    if window == 'hann':
        fade_window = np.hanning(fade_samples * 2)
    elif window == 'linear':
        fade_window = np.linspace(0.0, 1.0, fade_samples * 2)
    else:
        raise ValueError("window must be 'hann' or 'linear'")

    # Split window into fade-in and fade-out
    fade_in = fade_window[:fade_samples]
    fade_out = fade_window[fade_samples:]

    # Start with first segment
    result = segs[0].copy()

    for i in range(1, len(segs)):
        current = segs[i]
        overlap = min(fade_samples, len(result), len(current))

        if overlap <= 0:
            # No overlap possible, just concatenate
            result = np.concatenate([result, current])
            continue

        # Extract overlapping parts
        prev_tail = result[-overlap:]
        curr_head = current[:overlap]

        # Apply fade-out to previous tail, fade-in to current head
        prev_tail_faded = prev_tail * fade_out[-overlap:]
        curr_head_faded = curr_head * fade_in[:overlap]

        # Sum them
        blended = prev_tail_faded + curr_head_faded

        # Replace the tail of result with blended
        result = np.concatenate([result[:-overlap], blended, current[overlap:]])

    return result


def overlap_add(tiles, hop_length, window=None):
    """Overlap-add 2D tiles along the time axis.

    Parameters
    ----------
    tiles : list of np.ndarray
        List of 2D spectrogram-like tiles (e.g., frequency x time).
    hop_length : int
        Number of samples to advance between tiles.
    window : np.ndarray or None
        Optional window to apply to each tile before overlap-add.

    Returns
    -------
    np.ndarray
        The overlapped and summed output.
    """
    if not tiles:
        raise ValueError("tiles list is empty")

    tile_shape = np.asarray(tiles[0]).shape
    if len(tile_shape) != 2:
        raise ValueError("Tiles must be 2D arrays")

    # Determine output length
    n_tiles = len(tiles)
    tile_width = tile_shape[1]
    output_length = (n_tiles - 1) * hop_length + tile_width

    output = np.zeros((tile_shape[0], output_length), dtype=np.float32)

    # Precompute window if not provided
    if window is None:
        window = np.ones(tile_width, dtype=np.float32)
    else:
        window = np.asarray(window, dtype=np.float32)
        if window.shape != (tile_width,):
            raise ValueError("window shape must match tile width")

    for i, tile in enumerate(tiles):
        tile = np.asarray(tile, dtype=np.float32)
        if tile.shape != tile_shape:
            raise ValueError("All tiles must have the same shape")
        start = i * hop_length
        end = start + tile_width
        output[:, start:end] += tile * window[np.newaxis, :]

    return output
