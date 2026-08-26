import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff of a signal.

    The spectral rolloff is the frequency below which a given percentage
    (typically 85%) of the total spectral energy is contained. It is a
    useful descriptor for characterizing the brightness of a sound and is
    often used in music information retrieval and audio analysis.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz.
    rolloff_percent : float, optional
        Percentage of total spectral energy to consider (0.0 to 1.0).
        Default is 0.85.
    """

    def __init__(self, sample_rate=22050, rolloff_percent=0.85):
        self.sample_rate = sample_rate
        self.rolloff_percent = rolloff_percent

    def compute(self, spectrum):
        """Compute the spectral rolloff for a given magnitude spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            1D array of magnitude spectrum values (e.g., from an FFT).
            The array is assumed to be non-negative.

        Returns
        -------
        float
            The rolloff frequency in Hz.

        Raises
        ------
        ValueError
            If the spectrum is empty or contains negative values.
        """
        spectrum = np.asarray(spectrum, dtype=np.float64)
        if spectrum.ndim != 1 or spectrum.size == 0:
            raise ValueError("Spectrum must be a non-empty 1D array")
        if np.any(spectrum < 0):
            raise ValueError("Spectrum values must be non-negative")

        total_energy = np.sum(spectrum)
        if total_energy == 0:
            return 0.0

        cumulative = np.cumsum(spectrum)
        threshold = self.rolloff_percent * total_energy
        # Find the first index where cumulative energy exceeds threshold
        rolloff_index = np.searchsorted(cumulative, threshold)
        # Convert bin index to frequency
        # Frequency resolution is sample_rate / n_fft, but we don't have n_fft here.
        # Instead, assume spectrum is linearly spaced from 0 to Nyquist.
        n_bins = spectrum.size
        # The frequency of bin i is i * (sample_rate / 2) / (n_bins - 1)
        # But typically the spectrum is from 0 to sample_rate/2 with n_bins = n_fft/2+1
        # For simplicity, we map bin index to frequency assuming the last bin is Nyquist.
        max_freq = self.sample_rate / 2.0
        freq_res = max_freq / (n_bins - 1) if n_bins > 1 else 0.0
        rolloff_freq = rolloff_index * freq_res
        return float(rolloff_freq)
