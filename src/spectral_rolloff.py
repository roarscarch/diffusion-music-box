import numpy as np


def spectral_rolloff(spectrogram, sample_rate=22050, rolloff_percent=0.85):
    """Compute the spectral rolloff frequency for each frame of a spectrogram.

    The spectral rolloff is the frequency below which a given percentage of the
    total spectral energy is contained. It provides a measure of the spectral
    shape and can be used to characterize timbre or brightness of the audio.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (freq_bins, time_frames) containing magnitude or
        power values. Frequencies are assumed to be linearly spaced from 0 to
        Nyquist.
    sample_rate : int
        Sample rate of the audio in Hz.
    rolloff_percent : float, optional
        Percentage (0.0 to 1.0) of total spectral energy to reach the rolloff
        frequency. Default is 0.85.

    Returns
    -------
    np.ndarray
        1D array of length time_frames with the rolloff frequency in Hz for
        each frame.

    Raises
    ------
    ValueError
        If rolloff_percent is not between 0 and 1.
    """
    if not (0.0 < rolloff_percent < 1.0):
        raise ValueError("rolloff_percent must be between 0 and 1")

    n_freqs, n_frames = spectrogram.shape
    if n_freqs == 0:
        return np.zeros(n_frames)

    # Frequency axis (Hz) corresponding to each bin
    freqs = np.linspace(0, sample_rate / 2, n_freqs)

    # Cumulative sum along frequency axis
    total_energy = np.sum(spectrogram, axis=0)

    # Avoid division by zero for silent frames
    total_energy_safe = np.where(total_energy > 0, total_energy, 1.0)
    cumsum = np.cumsum(spectrogram, axis=0) / total_energy_safe

    # Find the first bin where cumulative sum exceeds rolloff_percent
    # For each frame, find the smallest index where cumsum >= rolloff_percent
    # Use argmax on a boolean mask to get the first True
    mask = cumsum >= rolloff_percent
    rolloff_indices = np.argmax(mask, axis=0)

    # For frames with zero energy, argmax returns 0, but we want 0 Hz
    # Set to 0 Hz for silent frames
    silent = total_energy <= 0
    rolloff_indices[silent] = 0

    return freqs[rolloff_indices]
