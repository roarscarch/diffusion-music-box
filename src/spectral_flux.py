import numpy as np


def spectral_flux(spectrogram, normalize=True):
    """Compute the spectral flux of a spectrogram.

    Spectral flux measures the rate of change in the magnitude spectrum
    between consecutive frames. It is commonly used for onset detection
    and to identify sudden changes in the audio signal.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (freq_bins, time_frames) containing magnitude
        spectrogram values (e.g., from STFT).
    normalize : bool, optional
        If True, normalize the flux values to the range [0, 1] by dividing
        by the maximum flux value (if non-zero). Default is True.

    Returns
    -------
    np.ndarray
        1D array of length (time_frames - 1) containing the spectral flux
        for each consecutive frame pair. If normalize is True, values are
        scaled to [0, 1].

    Raises
    ------
    ValueError
        If the spectrogram is not 2D.
    """
    if spectrogram.ndim != 2:
        raise ValueError(f"Spectrogram must be 2D, got {spectrogram.ndim}D")

    # Ensure float type
    spec = np.asarray(spectrogram, dtype=np.float64)

    # Compute difference between consecutive frames
    diff = np.diff(spec, axis=1)

    # Spectral flux is the sum of positive differences (or L2 norm)
    # Here we use L2 norm per frame for a smooth measure
    flux = np.sqrt(np.sum(diff ** 2, axis=0))

    if normalize:
        max_val = np.max(flux)
        if max_val > 0:
            flux = flux / max_val

    return flux
