import numpy as np


class SpectralFlux:
    """Compute spectral flux for onset and change detection.

    Spectral flux measures the change in magnitude spectrum between consecutive
    frames. It is commonly used for onset detection, beat tracking, and
    identifying abrupt changes in the audio signal. This implementation
    computes the L2 norm of the difference between successive spectrogram
    frames, optionally half-wave rectified (only positive changes).

    Parameters
    ----------
    frame_rate : int, optional
        Frame rate in frames per second (used for time scaling, not required).
    half_wave_rectify : bool, optional
        If True, only positive changes are considered (default True).
    """

    def __init__(self, frame_rate=1, half_wave_rectify=True):
        self.frame_rate = frame_rate
        self.half_wave_rectify = half_wave_rectify

    def compute(self, spectrogram):
        """Compute spectral flux for each frame.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, n_frames) representing magnitude
            spectrum (non-negative values).

        Returns
        -------
        np.ndarray
            1D array of length n_frames (first frame is 0) containing the
            spectral flux values.
        """
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if spectrogram.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)

        # Compute difference between consecutive frames
        diff = np.diff(spectrogram, axis=1)
        if self.half_wave_rectify:
            diff = np.maximum(diff, 0)
        # Compute L2 norm per frame (over frequency bins)
        flux = np.sqrt(np.sum(diff ** 2, axis=0))
        # Prepend zero for first frame
        flux = np.concatenate([[0.0], flux])
        return flux.astype(np.float32)

    def detect_onsets(self, spectrogram, threshold=0.5, min_frames=1):
        """Detect onset frames based on spectral flux.

        Onset frames are detected where the spectral flux exceeds a threshold
        (relative to the maximum flux in the spectrogram) and are separated by
        at least `min_frames` frames.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, n_frames).
        threshold : float, optional
            Relative threshold (0.0 to 1.0) of the maximum flux to consider an
            onset. Default 0.5.
        min_frames : int, optional
            Minimum number of frames between onsets. Default 1.

        Returns
        -------
        list of int
            Frame indices where onsets occur.
        """
        flux = self.compute(spectrogram)
        if len(flux) == 0:
            return []
        max_flux = np.max(flux)
        if max_flux == 0:
            return []
        abs_threshold = threshold * max_flux
        candidate_frames = np.where(flux > abs_threshold)[0].tolist()
        if not candidate_frames:
            return []

        # Enforce minimum distance between onsets
        onsets = [candidate_frames[0]]
        for frame in candidate_frames[1:]:
            if frame - onsets[-1] >= min_frames:
                onsets.append(frame)
        return onsets
