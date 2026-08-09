import numpy as np
import time
import threading
from typing import Callable, List, Tuple


class TempoScheduler:
    """Schedules parameter changes at a musical tempo.

    This module provides a way to schedule automated parameter changes at
    regular intervals based on a tempo (beats per minute). It is useful for
    creating evolving ambient music where parameters like noise schedule or
    diffusion steps change over time in a rhythmic manner.

    The scheduler maintains a list of scheduled events, each with a callback
    function and a beat interval. It runs in a background thread and invokes
    callbacks at the appropriate times.

    Parameters
    ----------
    bpm : float
        Beats per minute for the musical tempo.
    beats_per_bar : int, optional
        Number of beats per bar (default 4).
    """

    def __init__(self, bpm: float = 60.0, beats_per_bar: int = 4):
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self._events: List[Tuple[int, Callable, float]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._start_time = 0.0

    def set_tempo(self, bpm: float):
        """Update the tempo.

        Parameters
        ----------
        bpm : float
            New beats per minute value.
        """
        if bpm <= 0:
            raise ValueError("BPM must be positive")
        self.bpm = bpm

    def add_event(self, callback: Callable, interval_beats: float = 1.0, phase: float = 0.0):
        """Add a recurring event that fires every `interval_beats` beats.

        Parameters
        ----------
        callback : Callable
            Function to call when the event fires. No arguments are passed.
        interval_beats : float, optional
            Interval in beats between invocations (default 1).
        phase : float, optional
            Initial offset in beats from the start (default 0).
        """
        if interval_beats <= 0:
            raise ValueError("Interval must be positive")
        with self._lock:
            self._events.append((interval_beats, callback, phase))

    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the scheduler and wait for the thread to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self):
        """Background loop that fires events at the correct times."""
        while self._running:
            now = time.monotonic() - self._start_time
            beat_duration = 60.0 / self.bpm
            with self._lock:
                events = list(self._events)
            for interval, callback, phase in events:
                # Calculate the number of beats elapsed since start
                beats_elapsed = now / beat_duration
                # Check if an event should fire: (beats_elapsed - phase) is a multiple of interval
                if beats_elapsed >= phase:
                    # Determine the last fire time in beats
                    last_fire = phase + ((beats_elapsed - phase) // interval) * interval
                    # Fire if we're within a small window (e.g., 10ms) of the last fire time
                    if abs(beats_elapsed - last_fire) < 0.005:
                        callback()
            time.sleep(0.01)

    def reset(self):
        """Reset the scheduler to its initial state."""
        self.stop()
        with self._lock:
            self._events.clear()

    def get_elapsed_beats(self) -> float:
        """Return the number of beats that have elapsed since start."""
        if not self._running:
            return 0.0
        now = time.monotonic() - self._start_time
        return now / (60.0 / self.bpm)


# Convenience factory for common tempo-synchronized parameter changes
def create_tempo_synced_scheduler(bpm: float, callback: Callable, interval_beats: float = 1.0) -> TempoScheduler:
    """Create a TempoScheduler with a single recurring event.

    Parameters
    ----------
    bpm : float
        Beats per minute.
    callback : Callable
        Function to call on each beat.
    interval_beats : float, optional
        Interval in beats between calls.

    Returns
    -------
    TempoScheduler
        A configured scheduler instance.
    """
    scheduler = TempoScheduler(bpm=bpm)
    scheduler.add_event(callback, interval_beats=interval_beats)
    return scheduler
