import time
import threading


class MidiClock:
    """A MIDI clock for synchronizing generated audio with external devices.

    This module implements a simple MIDI clock that can be used to pace the
    generation of audio segments according to a tempo (in BPM). It provides
    a thread-safe way to query the current beat position and to wait for the
    next beat. The clock can be started, stopped, and reset, and it supports
    a callback that is invoked on each beat.

    Parameters
    ----------
    bpm : float
        Tempo in beats per minute.
    beats_per_bar : int, optional
        Number of beats per bar (default 4).
    """

    def __init__(self, bpm=120.0, beats_per_bar=4):
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self._beat_interval = 60.0 / bpm
        self._start_time = None
        self._beat_count = 0
        self._lock = threading.Lock()
        self._running = False
        self._beat_callback = None

    @property
    def beat_interval(self):
        """Return the time interval between beats in seconds."""
        return self._beat_interval

    def set_bpm(self, bpm):
        """Update the tempo.

        This updates the beat interval and, if the clock is running,
        recalculates the start time to keep the phase continuous.

        Parameters
        ----------
        bpm : float
            New tempo in beats per minute.
        """
        with self._lock:
            if bpm <= 0:
                raise ValueError("BPM must be positive")
            self.bpm = bpm
            new_interval = 60.0 / bpm
            if self._running and self._start_time is not None:
                # Keep the current beat phase by adjusting start time
                elapsed = time.time() - self._start_time
                beat_pos = elapsed / self._beat_interval
                self._start_time = time.time() - beat_pos * new_interval
            self._beat_interval = new_interval

    def start(self):
        """Start the clock."""
        with self._lock:
            if not self._running:
                self._start_time = time.time()
                self._running = True
                self._beat_count = 0

    def stop(self):
        """Stop the clock."""
        with self._lock:
            self._running = False
            self._start_time = None
            self._beat_count = 0

    def reset(self):
        """Reset the clock to zero and stop it."""
        with self._lock:
            self._running = False
            self._start_time = None
            self._beat_count = 0

    def is_running(self):
        """Return True if the clock is running."""
        with self._lock:
            return self._running

    def get_beat_count(self):
        """Return the current beat count (0-based)."""
        with self._lock:
            return self._beat_count

    def get_beat_position(self):
        """Return the current beat position as a float (fractional beats)."""
        with self._lock:
            if not self._running or self._start_time is None:
                return 0.0
            elapsed = time.time() - self._start_time
            return elapsed / self._beat_interval

    def wait_for_next_beat(self):
        """Block until the next beat boundary occurs.

        Returns
        -------
        int
            The beat count after the wait.
        """
        with self._lock:
            if not self._running:
                raise RuntimeError("Clock is not running")
            current_beat = self.get_beat_position()
            next_beat = int(current_beat) + 1
            wait_time = (next_beat - current_beat) * self._beat_interval
            # Release lock while sleeping
        time.sleep(wait_time)
        with self._lock:
            self._beat_count = int(self.get_beat_position())
            if self._beat_callback:
                self._beat_callback(self._beat_count)
            return self._beat_count

    def set_beat_callback(self, callback):
        """Set a callback to be invoked on each beat.

        Parameters
        ----------
        callback : callable
            Function taking a single integer beat count.
        """
        with self._lock:
            self._beat_callback = callback

    def get_next_beat_time(self):
        """Return the absolute time (seconds) of the next beat."""
        with self._lock:
            if not self._running or self._start_time is None:
                return None
            current = self.get_beat_position()
            next_beat = int(current) + 1
            return self._start_time + next_beat * self._beat_interval

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
}