import numpy as np
import time
import threading


class Metronome:
    """Generates a steady click track for tempo reference.

    The metronome produces a short percussive click at a configurable BPM.
    It runs in its own thread and can be started, stopped, and have its
    tempo changed live. The click is synthesized as a short burst of noise
    with an exponential decay, which cuts through ambient textures.

    Parameters
    ----------
    sample_rate : int
        Sample rate for the click sound.
    bpm : float
        Beats per minute.
    click_duration : float, optional
        Duration of each click in seconds. Defaults to 0.05.
    volume : float, optional
        Peak amplitude of the click. Defaults to 0.5.
    """

    def __init__(self, sample_rate=22050, bpm=60.0, click_duration=0.05, volume=0.5):
        self.sample_rate = sample_rate
        self.bpm = bpm
        self.click_duration = click_duration
        self.volume = volume
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._click = self._make_click()

    def _make_click(self):
        """Synthesize a short percussive click."""
        n = int(self.sample_rate * self.click_duration)
        t = np.arange(n) / self.sample_rate
        # Exponential decay envelope
        envelope = np.exp(-t * 80.0)
        # Noise burst with high-pass feel
        noise = np.random.default_rng(0).standard_normal(n)
        click = noise * envelope
        # Simple high-pass filter to remove rumble
        click = np.diff(click, prepend=0)
        return click * self.volume

    def set_bpm(self, bpm):
        """Update the tempo."""
        with self._lock:
            self.bpm = float(bpm)

    def get_bpm(self):
        """Return the current tempo."""
        with self._lock:
            return self.bpm

    def start(self):
        """Start the metronome thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the metronome thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def _run(self):
        """Play clicks at the configured BPM."""
        import sounddevice as sd
        try:
            sd.play(self._click, self.sample_rate)
        except Exception:
            # If sounddevice is not available or fails, just skip playback
            pass
        while self._running:
            bpm = self.get_bpm()
            if bpm <= 0:
                time.sleep(0.1)
                continue
            interval = 60.0 / bpm
            # Sleep in small increments to react to stop quickly
            slept = 0.0
            while self._running and slept < interval:
                time.sleep(0.01)
                slept += 0.01
            if self._running:
                try:
                    sd.play(self._click, self.sample_rate)
                except Exception:
                    pass
