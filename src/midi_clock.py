import threading
import time
import numpy as np


class MidiClock:
    """A simple MIDI clock generator for synchronizing audio or visual elements.

    This module provides a thread-safe clock that can send MIDI clock messages
    (24 pulses per quarter note) at a given tempo. It is designed to be used
    alongside the audio engine and other components to keep them in sync.
    The clock runs in its own thread and can be started, stopped, and have its
    tempo changed dynamically.
    """

    def __init__(self, bpm=120, pulses_per_beat=24):
        self.bpm = bpm
        self.pulses_per_beat = pulses_per_beat
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._next_pulse_time = 0.0
        self._pulse_count = 0
        self._listeners = []

    @property
    def pulse_interval(self):
        """Time in seconds between each clock pulse."""
        return 60.0 / (self.bpm * self.pulses_per_beat)

    def add_listener(self, callback):
        """Register a callback to be invoked on each pulse.

        Parameters
        ----------
        callback : callable
            A function that takes a single argument (pulse count) and returns None.
        """
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback):
        """Unregister a previously added listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def start(self):
        """Start the MIDI clock in a background thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._pulse_count = 0
            self._next_pulse_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the MIDI clock and wait for the thread to finish."""
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def set_bpm(self, bpm):
        """Change the tempo dynamically.

        Parameters
        ----------
        bpm : float
            Beats per minute. Must be > 0.
        """
        if bpm <= 0:
            raise ValueError("BPM must be positive")
        with self._lock:
            self.bpm = bpm
            # Recalculate next pulse time to avoid abrupt jumps
            self._next_pulse_time = time.monotonic() + self.pulse_interval

    def _run(self):
        """Internal loop that sends pulses at regular intervals."""
        while True:
            with self._lock:
                if not self._running:
                    break
                next_time = self._next_pulse_time
                pulse_count = self._pulse_count
                listeners = list(self._listeners)
                self._pulse_count += 1
                self._next_pulse_time = next_time + self.pulse_interval

            # Wait until the scheduled pulse time
            now = time.monotonic()
            if next_time > now:
                time.sleep(next_time - now)

            # Notify listeners outside the lock
            for callback in listeners:
                try:
                    callback(pulse_count)
                except Exception as e:
                    # Fail silently but report to stderr
                    import sys
                    print(f"Error in MIDI clock listener: {e}", file=sys.stderr)

            # If we fell behind, catch up by skipping missed pulses
            with self._lock:
                if self._running and self._next_pulse_time < time.monotonic():
                    self._next_pulse_time = time.monotonic() + self.pulse_interval

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_current_pulse(self):
        """Return the current pulse count (0-based)."""
        with self._lock:
            return self._pulse_count

    def get_current_beat(self):
        """Return the current beat number (0-based, each beat has pulses_per_beat pulses)."""
        with self._lock:
            return self._pulse_count // self.pulses_per_beat
