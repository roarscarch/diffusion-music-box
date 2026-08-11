import numpy as np


class OverlapAdd:
    """Synthesize time-domain audio from overlapping spectrogram tiles.

    This module converts a sequence of spectrogram tiles (frequency x time)
    into a continuous audio signal using the overlap-add method. It applies
    an inverse FFT to each tile's frames, windows them to reduce spectral
    leakage, and sums overlapping frames with proper normalization to avoid
    artifacts. This is a core building block for real-time generation in
    the diffusion music box.

    Parameters
    ----------
    fft_size : int
        FFT size used for the spectrogram (must be even).
    hop_length : int
        Hop length in samples between consecutive frames.
    window : str, optional
        Window function to apply to each frame. Options: 'hann', 'hamming',
        'blackman', 'rect'. Default is 'hann'.
    """

    def __init__(self, fft_size=1024, hop_length=256, window='hann'):
        if fft_size % 2 != 0:
            raise ValueError("fft_size must be even")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.num_freq_bins = fft_size // 2 + 1
        self.window = self._create_window(window)
        # Normalization factor to compensate for overlap
        self._norm_factor = self._compute_norm_factor()

    def _create_window(self, window_type):
        """Create a window function of length fft_size."""
        n = self.fft_size
        if window_type == 'hann':
            return np.hanning(n).astype(np.float32)
        elif window_type == 'hamming':
            return np.hamming(n).astype(np.float32)
        elif window_type == 'blackman':
            return np.blackman(n).astype(np.float32)
        elif window_type == 'rect':
            return np.ones(n, dtype=np.float32)
        else:
            raise ValueError(f"Unknown window type: {window_type}")

    def _compute_norm_factor(self):
        """Compute the normalization factor for overlap-add.

        The factor is the reciprocal of the sum of squared windows at each
        sample position, which ensures that overlapping frames sum to unity
        when using a window with the constant-overlap-add property.
        """
        # Build a long buffer to simulate the overlap-add of a constant signal
        length = self.fft_size * 4
        acc = np.zeros(length, dtype=np.float32)
        for start in range(0, length - self.fft_size + 1, self.hop_length):
            acc[start:start + self.fft_size] += self.window ** 2
        # Find the maximum accumulation to normalize (avoid division by zero)
        max_val = np.max(acc)
        if max_val > 0:
            return 1.0 / max_val
        else:
            return 1.0

    def _inverse_spectrogram(self, tile):
        """Convert a spectrogram tile to a time-domain signal.

        Parameters
        ----------
        tile : np.ndarray
            Spectrogram tile of shape (num_freq_bins, num_frames).
            The values are assumed to be magnitudes (non-negative).

        Returns
        -------
        np.ndarray
            Reconstructed time-domain signal of shape (num_frames * hop_length,).
        """
        num_frames = tile.shape[1]
        # Create a complex spectrum with random phase (for ambient texture)
        # Using a fixed seed for reproducibility? No, we want variation.
        # Use a deterministic phase based on the magnitude to avoid clicks.
        # Actually, for ambient music we can use a random phase that is smoothed.
        # Here we use a simple approach: zero phase, but that may sound dull.
        # Alternative: use the tile as magnitude and apply random phase.
        # For the overlap-add module, we'll treat the tile as the full
        # complex spectrum for now, but the diffusion model outputs magnitude.
        # So we create a complex spectrum with magnitude and random phase.
        # To keep it simple and avoid artifacts, we use a constant phase of 0
        # but that would produce symmetric signals. Instead, we use a phase
        # that is random per frame but consistent across time to avoid clicks.
        # We'll generate a random phase per frequency bin and reuse it across frames.
        rng = np.random.default_rng(42)  # fixed seed for determinism in tests
        phase = rng.uniform(-np.pi, np.pi, size=(self.num_freq_bins, 1))
        # Make phase smooth across time? Not necessary for now.
        # Build complex spectrum: magnitude * exp(i*phase), repeated for each frame
        spectrum = np.zeros((self.fft_size, num_frames), dtype=np.complex64)
        spectrum[:self.num_freq_bins, :] = tile * np.exp(1j * phase)
        # Mirror for negative frequencies (real signal)
        spectrum[self.num_freq_bins:] = np.conj(spectrum[1:self.fft_size - self.num_freq_bins + 1][::-1])
        # Actually, standard: for real FFT, bins 1..N-1 are conjugate symmetric.
        # We'll handle that properly below.
        # Simpler: use np.fft.irfft which expects only the positive frequencies.
        # Let's do that.
        # Positive frequencies: bins 0..num_freq_bins-1
        positive = tile * np.exp(1j * phase)  # shape (num_freq_bins, num_frames)
        # irfft expects shape (num_frames, num_freq_bins) but we'll transpose
        frames = np.fft.irfft(positive.T, n=self.fft_size, axis=1)  # shape (num_frames, fft_size)
        return frames

    def synthesize(self, tiles, overlap=0.5):
        """Synthesize continuous audio from a sequence of spectrogram tiles.

        Parameters
        ----------
        tiles : list of np.ndarray
            List of spectrogram tiles, each of shape (num_freq_bins, num_frames).
            All tiles must have the same number of frequency bins.
        overlap : float, optional
            Fraction of overlap between consecutive tiles (0.0 to 0.9).
            Default is 0.5, meaning half of each tile overlaps with the next.

        Returns
        -------
        np.ndarray
            Synthesized audio signal as a 1D float32 array.
        """
        if not tiles:
            return np.zeros(0, dtype=np.float32)
        # Validate shapes
        for tile in tiles:
            if tile.shape[0] != self.num_freq_bins:
                raise ValueError(f"Tile has {tile.shape[0]} frequency bins, expected {self.num_freq_bins}