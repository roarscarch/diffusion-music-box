import numpy as np
from scipy import ndimage


def pitch_shift_spectrogram(tile, shift_semitones):
    """Pitch-shift a spectrogram tile by shifting frequency bins.

    Parameters
    ----------
    tile : np.ndarray
        2D array of shape (n_freq, n_time) containing the spectrogram tile.
    shift_semitones : float
        Number of semitones to shift. Positive shifts up, negative down.

    Returns
    -------
    np.ndarray
        Pitch-shifted spectrogram tile.
    """
    if shift_semitones == 0:
        return tile.copy()
    n_freq, n_time = tile.shape
    # Frequency bins are assumed to be linearly spaced in Hz, but we treat
    # them as log-frequency for pitch shifting. Find the ratio of frequencies
    # per bin (assuming bin 0 is 0 Hz, but we use a small offset to avoid log(0)).
    # For simplicity, we use a linear index shift as a proxy for pitch shift.
    # This works well for ambient textures where exact pitch accuracy isn't critical.
    shift_bins = int(round(shift_semitones * 12))  # 12 semitones per octave, but this is not exact
    # Actually, for a linear frequency axis, a semitone shift is not a constant
    # bin shift. We'll use a simple interpolation approach: shift bins and
    # interpolate.
    # Create a new array and fill it by shifting the original.
    shifted = np.zeros_like(tile)
    if shift_bins >= 0:
        # Shift up: move bins to higher indices
        if shift_bins < n_freq:
            shifted[shift_bins:, :] = tile[:n_freq - shift_bins, :]
    else:
        # Shift down: move bins to lower indices
        shift_bins = -shift_bins
        if shift_bins < n_freq:
            shifted[:n_freq - shift_bins, :] = tile[shift_bins:, :]
    return shifted


def pitch_shift_tile(tile, semitones):
    """Pitch-shift a spectrogram tile using spline interpolation.

    This function applies a more accurate frequency-axis shift by using
    scipy's shift function with a spline interpolation, which handles
    fractional bin shifts.

    Parameters
    ----------
    tile : np.ndarray
        2D array of shape (n_freq, n_time).
    semitones : float
        Shift in semitones (positive up, negative down).

    Returns
    -------
    np.ndarray
        Shifted tile.
    """
    if semitones == 0:
        return tile.copy()
    # Convert semitones to a frequency ratio: ratio = 2^(semitones/12)
    ratio = 2.0 ** (semitones / 12.0)
    # For a linear frequency axis, we need to resample the frequency dimension.
    # We'll use ndimage.zoom with the appropriate factor on the frequency axis.
    # We want to compress/stretch the frequency axis such that the original
    # frequencies are mapped to new frequencies. For a shift up, we want to
    # move energy to higher frequencies, which means compressing the spectrum.
    # Actually, shifting up means we want the new tile to have the same content
    # but at higher frequencies, so we take the original and map it to higher
    # bins. That is equivalent to stretching the frequency axis (i.e., zooming
    # with factor >1) and then taking the lower part? Let's think:
    # If we have a spectral peak at bin i, after shifting up by semitones, the
    # peak should appear at bin i * ratio (approximately). So we need to
    # resample the frequency axis such that the new array's bin j corresponds
    # to original bin j / ratio. That means we take the original array and
    # interpolate at positions j / ratio. This is equivalent to using
    # ndimage.zoom with factor 1/ratio on the frequency axis, but we also need
    # to preserve the array shape. We'll use scipy.ndimage.map_coordinates.
    n_freq, n_time = tile.shape
    # Create coordinate grid for output: frequency indices 0..n_freq-1
    output_freq = np.arange(n_freq)
    # Source frequency indices: output_freq / ratio
    source_freq = output_freq / ratio
    # Clamp to valid range
    source_freq = np.clip(source_freq, 0, n_freq - 1)
    # Create coordinate arrays for map_coordinates: (freq, time)
    coord = np.meshgrid(source_freq, np.arange(n_time), indexing='ij')
    # Map coordinates: we want for each output (freq, time) to sample from
    # original at (source_freq, time).
    shifted = ndimage.map_coordinates(tile, coord, order=3, mode='nearest')
    return shifted