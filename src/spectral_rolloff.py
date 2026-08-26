import numpy as np


class SpectralRolloff:
    """Compute the spectral rolloff point of a signal.

    The spectral rolloff is the frequency below which a specified percentage
    (usually 85% or 95%) of the total spectral energy is contained. It is a
    useful feature for distinguishing between different timbres and for
    detecting the brightness of a sound. This module provides both a simple
    function and a class-based interface for streaming or batch analysis.
    """

    def __init__(self, sample_rate=22050, percentile=0.85):
        """Initialize the spectral rolloff calculator.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the audio signal in Hz.
        percentile : float
            Fraction of total energy used as the threshold (between 0 and 1).
        """
        self.sample_rate = sample_rate
        self.percentile = percentile

    def compute(self, spectrum, frequencies=None):
        """Compute the spectral rolloff frequency from a magnitude spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            1D array of magnitude spectrum values (e.g., from an FFT).
        frequencies : np.ndarray, optional
            1D array of frequencies corresponding to each bin. If None,
            uses bin indices scaled by sample_rate / (2 * (len(spectrum)-1)).

        Returns
        -------
        float
            The rolloff frequency in Hz.
        """
        spectrum = np.asarray(spectrum, dtype=np.float64)
        if spectrum.ndim != 1:
            raise ValueError("Spectrum must be 1D")
        if len(spectrum) == 0:
            return 0.0

        total_energy = np.sum(spectrum)
        if total_energy <= 0:
            return 0.0

        if frequencies is None:
            # Assume spectrum is from FFT with symmetric bins
            n_bins = len(spectrum)
            nyquist = self.sample_rate / 2.0
            frequencies = np.linspace(0.0, nyquist, n_bins)
        else:
            frequencies = np.asarray(frequencies, dtype=np.float64)
            if frequencies.shape != spectrum.shape:
                raise ValueError("Frequencies must have same shape as spectrum")

        # Compute cumulative energy and find bin where threshold is crossed
        cumulative = np.cumsum(spectrum)
        threshold = self.percentile * total_energy
        # Find first index where cumulative >= threshold
        idx = np.searchsorted(cumulative, threshold)
        idx = min(idx, len(spectrum) - 1)
        return float(frequencies[idx])

    def compute_spectrogram_rolloff(self, spectrogram):
        """Compute rolloff for each frame of a spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (n_freq_bins, n_frames) or (n_frames, n_freq_bins).
            This implementation expects frequency bins along axis 0.

        Returns
        -------
        np.ndarray
            1D array of rolloff frequencies per frame (length n_frames).
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float64)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")

        n_freq_bins = spectrogram.shape[0]
        nyquist = self.sample_rate / 2.0
        frequencies = np.linspace(0.0, nyquist, n_freq_bins)

        rolloffs = np.zeros(spectrogram.shape[1], dtype=np.float64)
        for i in range(spectrogram.shape[1]):
            rolloffs[i] = self.compute(spectrogram[:, i], frequencies)
        return rolloffs

    def __call__(self, spectrum, frequencies=None):
        """Shorthand for compute()."""
        return self.compute(spectrum, frequencies)


def compute_spectral_rolloff(spectrum, sample_rate=22050, percentile=0.85, frequencies=None):
    """Compute the spectral rolloff of a magnitude spectrum.

    This is a convenience function that creates a SpectralRolloff instance and
    calls its compute method.

    Parameters
    ----------
    spectrum : np.ndarray
        1D magnitude spectrum.
    sample_rate : int
        Sample rate in Hz.
    percentile : float
        Energy fraction threshold.
    frequencies : np.ndarray, optional
        Frequency values for each bin.

    Returns
    -------
    float
        Rolloff frequency in Hz.
    """
    rolloff_calc = SpectralRolloff(sample_rate=sample_rate, percentile=percentile)
    return rolloff_calc.compute(spectrum, frequencies)
