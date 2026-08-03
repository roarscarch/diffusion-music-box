import numpy as np
import threading
import time


class SpectrumVisualizer:
    """Displays a real-time spectrum visualization in the terminal.

    This module provides a simple ASCII-based visualization of the audio
    spectrum for the given audio buffer. It runs in a separate thread and
    updates the display at a configurable refresh rate.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio data.
    block_size : int
        Number of samples per block.
    refresh_rate : float
        Refresh rate in Hz for the visualization.
    n_bars : int
        Number of bars to display in the ASCII spectrum.
    """

    def __init__(self, sample_rate=22050, block_size=1024, refresh_rate=10.0, n_bars=40):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.refresh_rate = refresh_rate
        self.n_bars = n_bars

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._audio_buffer = np.zeros(block_size * 8, dtype=np.float32)

        # Frequencies for the bars (log-spaced)
        self._freq_edges = np.logspace(
            np.log10(20.0),
            np.log10(min(self.sample_rate / 2, 20000.0)),
            self.n_bars + 1
        )
        self._freq_centers = 0.5 * (self._freq_edges[:-1] + self._freq_edges[1:])

    def update_audio(self, audio):
        """Update the audio buffer for visualization.

        Parameters
        ----------
        audio : np.ndarray
            New audio samples to add to the visualization buffer.
        """
        if audio is None or len(audio) == 0:
            return
        with self._lock:
            # Shift the buffer and append new samples
            shift = min(len(audio), len(self._audio_buffer))
            self._audio_buffer[:-shift] = self._audio_buffer[shift:]
            self._audio_buffer[-shift:] = audio[-shift:]

    def start(self):
        """Start the visualization thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the visualization thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self):
        """Main loop for the visualization."""
        while self._running:
            self._draw()
            time.sleep(1.0 / self.refresh_rate)

    def _compute_spectrum(self):
        """Compute the magnitude spectrum of the current buffer.

        Returns
        -------
        np.ndarray
            Array of shape (n_bars,) with magnitudes in dB normalized to [0, 1].
        """
        with self._lock:
            data = self._audio_buffer.copy()
        if len(data) < 2:
            return np.zeros(self.n_bars)

        # Apply a Hann window to reduce spectral leakage
        window = np.hanning(len(data))
        data = data * window

        # Compute FFT
        spectrum = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1.0 / self.sample_rate)

        # Map to bars
        magnitudes = np.zeros(self.n_bars)
        for i in range(self.n_bars):
            lo = self._freq_edges[i]
            hi = self._freq_edges[i + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                magnitudes[i] = np.mean(spectrum[mask])

        # Convert to dB and normalize
        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(magnitudes + 1e-10)
        db_min = -80.0
        db_max = 0.0
        normalized = (db - db_min) / (db_max - db_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized

    def _draw(self):
        """Draw the ASCII spectrum to stdout."""
        spectrum = self._compute_spectrum()
        bars = [int(v * 20.0) for v in spectrum]
        line = "".join("#" * b + " " * (20 - b) for b in bars)
        # Clear the current line and print
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def __del__(self):
        self.stop()
