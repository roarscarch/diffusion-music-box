import time
import threading


class MidiClock:
    """A simple MIDI clock for tempo-synced diffusion generation.

    This module provides a reference clock that can be used to schedule
    generation steps or note events in sync with a musical tempo. It
    supports setting a BPM and provides methods to get the current beat
    and time until the next beat.

    Parameters
    ----------
    bpm : float, optional
        Initial tempo in beats per minute.
    """

    def __init__(self, bpm=120.0):
        self.bpm = bpm
        self._start_time = time.monotonic()
        self._lock = threading.Lock()

    @property
    def beat_duration(self):
        """Duration of one beat in seconds."""
        return 60.0 / self.bpm

    def set_bpm(self, bpm):
        """Update the tempo.

        Parameters
        ----------
        bpm : float
            New tempo in beats per minute. Must be positive.
        """
        if bpm <= 0:
            raise ValueError("BPM must be positive")
        with self._lock:
            # Align current beat position to avoid jumps
            current_beat = self.get_beat()
            self.bpm = bpm
            self._start_time = time.monotonic() - current_beat * self.beat_duration

    def get_beat(self):
        """Current beat position (float) since clock start."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return elapsed / self.beat_duration

    def get_beat_count(self):
        """Integer beat count (floor of current beat)."""
        return int(self.get_beat())

    def time_until_next_beat(self):
        """Seconds until the next beat boundary."""
        beat = self.get_beat()
        frac = beat - int(beat)
        return (1.0 - frac) * self.beat_duration

    def reset(self):
        """Reset the clock to beat zero."""
        with self._lock:
            self._start_time = time.monotonic()

    def wait_for_beat(self, beat_count=None):
        """Block until the next beat (or a specific beat count).

        Parameters
        ----------
        beat_count : int, optional
            If given, wait until this specific beat number. Otherwise wait
            for the next beat boundary.

        Returns
        -------
        int
            The beat count that was reached.
        """
        if beat_count is None:
            beat_count = self.get_beat_count() + 1
        while self.get_beat_count() < beat_count:
            time.sleep(0.001)
        return beat_count
