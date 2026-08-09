import threading
import time


class MidiClockSync:
    """Synchronize the music generation tempo with an external MIDI clock.

    This module listens for MIDI clock messages (typically 24 pulses per
    quarter note) and provides a tempo estimate that can be used to update
    the generation scheduler in real time. It supports starting/stopping
    the clock and resetting the beat position.

    Parameters
    ----------
    ppqn : int, optional
        Pulses per quarter note (standard MIDI clock is 24).
    smoothing : float, optional
        Exponential smoothing factor for tempo updates (0.0 to 1.0).
        Higher values give more weight to recent measurements.
    """

    def __init__(self, ppqn=24, smoothing=0.2):
        self.ppqn = ppqn
        self.smoothing = smoothing
        self._lock = threading.Lock()
        self._running = False
        self._last_pulse_time = None
        self._tempo = None  # beats per minute
        self._pulses_since_start = 0
        self._start_time = None

    def start(self):
        """Start or restart the MIDI clock."""
        with self._lock:
            self._running = True
            self._last_pulse_time = None
            self._tempo = None
            self._pulses_since_start = 0
            self._start_time = time.time()

    def stop(self):
        """Stop the MIDI clock and clear tempo state."""
        with self._lock:
            self._running = False
            self._tempo = None
            self._last_pulse_time = None
            self._pulses_since_start = 0
            self._start_time = None

    def reset(self):
        """Reset the beat position without stopping the clock."""
        with self._lock:
            self._pulses_since_start = 0
            self._start_time = time.time()
            self._last_pulse_time = None

    def pulse(self):
        """Process a single MIDI clock pulse."""
        now = time.time()
        with self._lock:
            if not self._running:
                return
            if self._last_pulse_time is not None:
                delta = now - self._last_pulse_time
                if delta > 0:
                    # Tempo in BPM = 60 / (delta * ppqn)
                    instant_bpm = 60.0 / (delta * self.ppqn)
                    if self._tempo is None:
                        self._tempo = instant_bpm
                    else:
                        self._tempo = (self.smoothing * instant_bpm +
                                       (1 - self.smoothing) * self._tempo)
            self._last_pulse_time = now
            self._pulses_since_start += 1

    def get_tempo(self):
        """Return the current tempo estimate in BPM.

        Returns
        -------
        float or None
            Estimated tempo in beats per minute, or None if not enough data.
        """
        with self._lock:
            return self._tempo

    def get_beat_position(self):
        """Return the current beat position since start/reset.

        Returns
        -------
        float
            Beat position in beats (fractional). Returns 0 if not running.
        """
        with self._lock:
            if not self._running or self._start_time is None:
                return 0.0
            elapsed = time.time() - self._start_time
            if self._tempo is None:
                return 0.0
            return elapsed * self._tempo / 60.0

    def is_running(self):
        """Return whether the MIDI clock is active."""
        with self._lock:
            return self._running

    def set_tempo(self, bpm):
        """Manually set the tempo (useful for testing or fallback)."""
        with self._lock:
            self._tempo = float(bpm)
