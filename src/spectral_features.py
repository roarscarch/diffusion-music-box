import numpy as np


def spectral_centroid(spectrogram, sample_rate=22050):
    """Compute the spectral centroid for each time frame.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (freq_bins, time_frames) containing magnitude values.
    sample_rate : int
        Sample rate of the audio.

    Returns
    -------
    np.ndarray
        1D array of centroid frequencies in Hz for each frame.
    """
    freq_bins, num_frames = spectrogram.shape
    freqs = np.linspace(0, sample_rate / 2, freq_bins)
    # Avoid division by zero
    total_energy = np.sum(spectrogram, axis=0)
    total_energy[total_energy == 0] = 1e-10
    centroid = np.sum(spectrogram * freqs[:, np.newaxis], axis=0) / total_energy
    return centroid


def spectral_rolloff(spectrogram, sample_rate=22050, rolloff_percent=0.85):
    """Compute the spectral rolloff frequency for each time frame.

    The rolloff frequency is the frequency below which a given percentage
    of the total spectral energy is contained.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (freq_bins, time_frames) containing magnitude values.
    sample_rate : int
        Sample rate of the audio.
    rolloff_percent : float
        Fraction of total energy (0.0 to 1.0) at which to find the rolloff.

    Returns
    -------
    np.ndarray
        1D array of rolloff frequencies in Hz for each frame.
    """
    freq_bins, num_frames = spectrogram.shape
    freqs = np.linspace(0, sample_rate / 2, freq_bins)
    cumsum = np.cumsum(spectrogram, axis=0)
    total_energy = cumsum[-1, :]
    total_energy[total_energy == 0] = 1e-10
    target = rolloff_percent * total_energy

    rolloff_indices = np.zeros(num_frames, dtype=int)
    for i in range(num_frames):
        idx = np.searchsorted(cumsum[:, i], target[i])
        rolloff_indices[i] = min(idx, freq_bins - 1)
    return freqs[rolloff_indices]
