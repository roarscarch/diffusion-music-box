import numpy as np


def spectral_rolloff(spectrogram, sample_rate, rolloff_percent=0.85):
    """Compute the spectral rolloff for each frame of a spectrogram.

    The spectral rolloff is the frequency below which a given percentage of
    the total spectral energy is contained. It is a useful feature for
    distinguishing between tonal and noisy signals, and can be used to
    analyze the brightness of the generated ambient music.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (freq_bins, time_frames) containing magnitudes or
        power values. Each column is a frame.
    sample_rate : int
        Sample rate of the audio used to produce the spectrogram.
    rolloff_percent : float, optional
        Percentage (0.0 to 1.0) of the total energy to include below the
        rolloff frequency. Default is 0.85.

    Returns
    -------
    np.ndarray
        1D array of length time_frames containing the rolloff frequency in Hz
        for each frame.

    Raises
    ------
    ValueError
        If rolloff_percent is not in the range (0, 1].
    """
    if not 0.0 < rolloff_percent <= 1.0:
        raise ValueError("rolloff_percent must be in the range (0, 1]")

    # Ensure input is a numpy array of floats
    spec = np.asarray(spectrogram, dtype=np.float64)
    if spec.ndim != 2:
        raise ValueError("spectrogram must be a 2D array")

    # Compute the cumulative sum of energy along the frequency axis
    cumulative_energy = np.cumsum(spec, axis=0)
    total_energy = cumulative_energy[-1, :]

    # Avoid division by zero by setting total energy to a small value where zero
    total_energy_safe = np.where(total_energy == 0, 1e-12, total_energy)
    normalized_energy = cumulative_energy / total_energy_safe

    # Find the index where the cumulative energy exceeds the rolloff percentage
    # For each frame, find the first index where normalized_energy >= rolloff_percent
    # Use argmax on a boolean mask; since True is 1 and False is 0, argmax returns the first True.
    # If no index satisfies, argmax returns 0, but we'll handle that case.
    mask = normalized_energy >= rolloff_percent
    rolloff_indices = np.argmax(mask, axis=0)

    # For frames where total energy is zero, rolloff is undefined; set to 0 Hz.
    zero_energy_frames = total_energy == 0
    rolloff_indices[zero_energy_frames] = 0

    # Convert bin index to frequency
    # The frequency for bin i is i * sample_rate / (2 * (freq_bins - 1)) or similar depending on FFT.
    # We assume the spectrogram is produced with a standard FFT where bin i corresponds to
    # frequency i * sample_rate / N, but N is not known. We can only compute a relative frequency
    # if we know the number of bins and the Nyquist frequency.
    # We'll assume the spectrogram covers frequencies from 0 to Nyquist (sample_rate/2).
    freq_bins = spec.shape[0]
    if freq_bins <= 1:
        return np.zeros(spec.shape[1], dtype=np.float64)
    # Frequency per bin: (sample_rate/2) / (freq_bins - 1) for bins 0..freq_bins-1
    freq_per_bin = (sample_rate / 2.0) / (freq_bins - 1)
    rolloff_freqs = rolloff_indices * freq_per_bin

    return rolloff_freqs
