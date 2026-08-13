import numpy as np
import threading
from midi_clock import MidiClock


class MidiController:
    """Handles MIDI-like keyboard input for interactive control.

    This class listens for keyboard events and translates them into
    parameter changes for the diffusion music box. It provides a
    simple event queue and a polling interface for the main loop.

    Parameters
    ----------
    clock : MidiClock
        The clock instance used for timing and synchronization.
    """

    def __init__(self, clock=None):
        self.clock = clock or MidiClock()
        self._event_queue = []
        self._lock = threading.Lock()
        self._running = False
        self._key_map = {
            'a': 'toggle_play',
            's': 'stop',
            'd': 'next_preset',
            'f': 'prev_preset',
            'j': 'tempo_down',
            'k': 'tempo_up',
            'l': 'toggle_crossfade',
            ';': 'toggle_arp',
            'q': 'quit',
        }

    def start(self):
        """Start listening for keyboard input."""
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening for keyboard input."""
        self._running = False

    def _listen(self):
        """Internal loop reading characters from stdin."""
        import sys
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                ch = sys.stdin.read(1)
                if ch:
                    self._handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key):
        """Process a single key press.

        Parameters
        ----------
        key : str
            The character read from stdin.
        """
        action = self._key_map.get(key.lower())
        if action:
            with self._lock:
                self._event_queue.append((action, self.clock.get_time()))

    def poll_events(self):
        """Retrieve and clear pending events.

        Returns
        -------
        list of tuple
            Each tuple is (action, timestamp).
        """
        with self._lock:
            events = list(self._event_queue)
            self._event_queue.clear()
            return events

    def get_state(self):
        """Return current controller state for integration.

        Returns
        -------
        dict
            A snapshot of relevant state.
        """
        return {
            'running': self._running,
            'key_map': dict(self._key_map),
            'clock_time': self.clock.get_time(),
        }
