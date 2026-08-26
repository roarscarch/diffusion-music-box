import numpy as np


def spectral_rolloff(spectrogram, sample_rate, rolloff_percent=0.85):
    """Compute the spectral rolloff frequency for each frame of a spectrogram.

    The spectral rolloff is the frequency below which a specified percentage
    (e.g., 85%) of the total spectral energy is contained. It is a measure of
    the spectral shape and can be used to characterize the brightness or
    sharpness of an audio signal.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (n_freq_bins, n_frames) representing magnitude
        spectrogram values.
    sample_rate : int
        Sample rate of the audio signal in Hz.
    rolloff_percent : float, optional
        Percentage of total energy below the rolloff frequency (0.0 to 1.0).
        Default is 0.85.

    Returns
    -------
    np.ndarray
        1D array of length n_frames with the rolloff frequency in Hz for each frame.
    """
    if not 0.0 < rolloff_percent <= 1.0:
        raise ValueError("rolloff_percent must be in (0, 1]")

    # Compute cumulative sum along frequency axis
    cumulative = np.cumsum(spectrogram, axis=0)
    total_energy = cumulative[-1, :]

    # Avoid division by zero for silent frames
    total_energy = np.maximum(total_energy, 1e-12)

    # Normalize cumulative sum to get fraction of total energy
    normalized = cumulative / total_energy[np.newaxis, :]

    # Find the first frequency bin where normalized sum exceeds rolloff_percent
    # For each frame, find the index of the first bin >= rolloff_percent
    # Use argmax on a boolean mask (first True) - but careful: if no bin exceeds, argmax returns 0
    # We'll handle by checking if any bin exceeds, else use last bin
    exceeds = normalized >= rolloff_percent
    rolloff_indices = np.argmax(exceeds, axis=0)

    # If a frame has no bin exceeding, argmax returns 0, which is wrong. Fix by setting to last bin.
    has_exceed = np.any(exceeds, axis=0)
    rolloff_indices[~has_exceed] = spectrogram.shape[0] - 1

    # Convert bin index to frequency: bin i corresponds to frequency (i * sample_rate / (2 * (n_bins - 1)))
    # Assuming spectrogram has n_bins = fft_size/2 + 1, so max frequency = sample_rate/2.
    n_bins = spectrogram.shape[0]
    freq_per_bin = sample_rate / (2.0 * (n_bins - 1))
    rolloff_freqs = rolloff_indices * freq_per_bin

    return rolloff_freqs
