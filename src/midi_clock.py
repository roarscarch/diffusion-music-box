class MIDIClock:
    """Provides MIDI clock synchronization and tempo control.

    This class simulates a MIDI clock for external synchronization and
    provides a simple way to track tempo and send clock pulses. It can be
    used to synchronize the ambient music generation with external devices
    or to maintain a steady internal tempo.

    Parameters
    ----------
    bpm : float
        Beats per minute for the clock.
    """

    def __init__(self, bpm=120.0):
        self.bpm = bpm
        self._pulse_count = 0
        self._running = False
        self._tick_time = 0.0

    def start(self):
        """Start the clock."""
        self._running = True
        self._pulse_count = 0

    def stop(self):
        """Stop the clock."""
        self._running = False

    def set_bpm(self, bpm):
        """Set the tempo.

        Parameters
        ----------
        bpm : float
            New beats per minute value.
        """
        if bpm <= 0:
            raise ValueError("BPM must be positive")
        self.bpm = bpm

    def tick(self, sample_rate=1):
        """Advance the clock by one tick (e.g., per audio block).

        Parameters
        ----------
        sample_rate : int
            Sample rate of audio, used to compute pulse interval in time.

        Returns
        -------
        bool
            True if a clock pulse (24 pulses per quarter note) should be sent.
        """
        if not self._running:
            return False

        # 24 pulses per quarter note (MIDI standard)
        pulses_per_beat = 24
        pulses_per_second = self.bpm * pulses_per_beat / 60.0
        pulse_interval = 1.0 / pulses_per_second

        # Increment time by one sample duration
        self._tick_time += 1.0 / sample_rate

        if self._tick_time >= pulse_interval:
            self._tick_time -= pulse_interval
            self._pulse_count += 1
            return True
        return False

    def get_pulse_count(self):
        """Return the total number of pulses sent since start."""
        return self._pulse_count

    def get_beat_count(self):
        """Return the number of quarter notes since start."""
        return self._pulse_count // 24

    def reset(self):
        """Reset the clock to initial state."""
        self._pulse_count = 0
        self._tick_time = 0.0
        self._running = False
