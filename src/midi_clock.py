import threading
import time
import numpy as np


class MidiClock:
    """A simple MIDI clock generator for tempo-synced diffusion generation.

    This class provides a beat-accurate clock that can be used to schedule
    generation steps in sync with a musical tempo. It supports starting,
    stopping, and querying the current beat position. The clock runs in a
    background thread and can be used by the diffusion model to generate
    segments at a steady tempo, enabling rhythmic and tempo-aligned ambient
    textures.

    Parameters
    ----------
    bpm : float, optional
        Beats per minute (default 120).
    beats_per_bar : int, optional
        Number of beats per bar (default 4).
    """

    def __init__(self, bpm=120.0, beats_per_bar=4):
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._beat = 0.0
        self._bar = 0
        self._tick_count = 0
        self._start_time = 0.0
        self._last_tick_time = 0.0
        self._listeners = []

    def start(self):
        """Start the MIDI clock."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._last_tick_time = self._start_time
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the MIDI clock."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self):
        """Main loop that fires ticks at the current tempo."""
        tick_interval = 60.0 / self.bpm / 24.0  # MIDI clock ticks per beat = 24
        while self._running:
            now = time.time()
            elapsed = now - self._start_time
            beat = elapsed * self.bpm / 60.0
            with self._lock:
                self._beat = beat
                self._bar = int(beat // self.beats_per_bar)
                self._tick_count = int(beat * 24.0)
            # Notify listeners at each tick
            if now - self._last_tick_time >= tick_interval:
                self._last_tick_time = now
                self._notify_tick()
            time.sleep(0.001)

    def _notify_tick(self):
        """Call all registered tick listeners."""
        for listener in self._listeners:
            try:
                listener(self._beat, self._bar)
            except Exception as e:
                print(f"MIDI clock listener error: {e}