import numpy as np


class SpectrumAutocorrelation:
    """Compute autocorrelation of spectrogram tiles for texture analysis.

    Autocorrelation provides a measure of self-similarity across time and
    frequency, which can be used to identify repetitive patterns in the
    generated ambient music. This module computes both time-domain and
    frequency-domain autocorrelation of a spectrogram tile, returning
    normalized correlation coefficients. The results can be used for
    features like detecting rhythmic stability or spectral periodicity.
    """

    def __init__(self, max_lag=None):
        """Initialize the autocorrelation module.

        Parameters
        ----------
        max_lag : int, optional
            Maximum lag in frames for time autocorrelation. If None,
            defaults to half the number of time frames.
        """
        self.max_lag = max_lag

    def time_autocorrelation(self, spectrogram):
        """Compute normalized autocorrelation along the time axis.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames).

        Returns
        -------
        np.ndarray
            1D array of autocorrelation coefficients for lags 0..max_lag,
            normalized so lag 0 equals 1.0.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        n_frames = spectrogram.shape[1]
        if n_frames < 2:
            return np.array([1.0])
        max_lag = self.max_lag if self.max_lag is not None else n_frames // 2
        max_lag = max(0, min(max_lag, n_frames - 1))

        # Center the data
        centered = spectrogram - spectrogram.mean(axis=1, keepdims=True)
        denom = np.sum(centered ** 2, axis=1) + 1e-10
        autocorr = np.zeros(max_lag + 1, dtype=np.float32)
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                num = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
                autocorr[lag] = np.mean(num / denom)
        return autocorr

    def frequency_autocorrelation(self, spectrogram):
        """Compute normalized autocorrelation along the frequency axis.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames).

        Returns
        -------
        np.ndarray
            1D array of autocorrelation coefficients for lags 0..max_lag,
            normalized so lag 0 equals 1.0.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        n_bins = spectrogram.shape[0]
        if n_bins < 2:
            return np.array([1.0])
        max_lag = self.max_lag if self.max_lag is not None else n_bins // 2
        max_lag = max(0, min(max_lag, n_bins - 1))

        # Center the data along frequency
        centered = spectrogram - spectrogram.mean(axis=0, keepdims=True)
        denom = np.sum(centered ** 2, axis=0) + 1e-10
        autocorr = np.zeros(max_lag + 1, dtype=np.float32)
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                num = np.sum(centered[:-lag, :] * centered[lag:, :], axis=0)
                autocorr[lag] = np.mean(num / denom)
        return autocorr

    def correlation_length(self, autocorr, threshold=0.5):
        """Estimate the correlation length from an autocorrelation curve.

        The correlation length is the smallest lag at which the autocorrelation
        drops below the given threshold (or 0 if never). This provides a
        simple summary statistic for texture analysis.

        Parameters
        ----------
        autocorr : np.ndarray
            1D array of autocorrelation coefficients (lag 0 first).
        threshold : float, optional
            Value below which the lag is considered decorrelated.

        Returns
        -------
        int
            Lag at which autocorrelation drops below threshold, or len(autocorr)-1 if never.
        """
        autocorr = np.asarray(autocorr)
        for lag, val in enumerate(autocorr):
            if val < threshold:
                return lag
        return len(autocorr) - 1

    def analyze(self, spectrogram):
        """Compute time and frequency autocorrelation together.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames).

        Returns
        -------
        dict
            Dictionary with keys 'time_autocorr', 'freq_autocorr',
            'time_corr_length', 'freq_corr_length'.
        """
        time_autocorr = self.time_autocorrelation(spectrogram)
        freq_autocorr = self.frequency_autocorrelation(spectrogram)
        return {
            'time_autocorr': time_autocorr,
            'freq_autocorr': freq_autocorr,
            'time_corr_length': self.correlation_length(time_autocorr),
            'freq_corr_length': self.correlation_length(freq_autocorr),
        }
