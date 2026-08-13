import numpy as np
import threading
import time


class MIDIController:
    """A simple MIDI-like controller interface for keyboard input.

    This module provides a headless CLI-based controller that maps keyboard
    presses to MIDI-like control messages. It allows interactive adjustment
    of parameters such as diffusion steps, noise schedule, and tempo.
    It runs in a background thread, listening for key presses and updating
    internal state that can be polled by other modules.
    """

    def __init__(self, poll_interval=0.05):
        """Initialize the controller.

        Parameters
        ----------
        poll_interval : float
            Seconds between key state polls. Lower values increase responsiveness.
        """
        self._poll_interval = poll_interval
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._key_states = {}
        self._last_key = None
        self._key_count = 0
        self._callback = None

    def start(self):
        """Start the keyboard listener thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the keyboard listener thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def set_callback(self, callback):
        """Set a callback for key press events.

        Parameters
        ----------
        callback : callable
            Function called with a key name string on each press.
        """
        with self._lock:
            self._callback = callback

    def _run(self):
        """Main loop for keyboard polling."""
        try:
            import termios
            import tty
            import sys
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except (ImportError, AttributeError, termios.error):
            # Fallback: no raw mode, just read lines
            old_settings = None

        try:
            while self._running:
                try:
                    import sys
                    import select
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1)
                        if key:
                            self._handle_key(key)
                except (IOError, ValueError):
                    pass
                time.sleep(self._poll_interval)
        finally:
            if old_settings is not None:
                import termios
                import sys
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)

    def _handle_key(self, key):
        """Process a single key press.

        Parameters
        ----------
        key : str
            The character pressed.
        """
        with self._lock:
            self._key_states[key] = time.time()
            self._last_key = key
            self._key_count += 1
            callback = self._callback
        if callback:
            callback(key)

    def get_key_state(self, key):
        """Return the timestamp of the last press of a key.

        Parameters
        ----------
        key : str
            The key to query.

        Returns
        -------
        float or None
            Time of last press, or None if never pressed.
        """
        with self._lock:
            return self._key_states.get(key)

    def get_last_key(self):
        """Return the most recently pressed key.

        Returns
        -------
        str or None
            The last key pressed, or None if none yet.
        """
        with self._lock:
            return self._last_key

    def get_key_count(self):
        """Return the total number of key presses registered.

        Returns
        -------
        int
            Total key press count.
        """
        with self._lock:
            return self._key_count

    def clear(self):
        """Reset key state and count."""
        with self._lock:
            self._key_states.clear()
            self._last_key = None
            self._key_count = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
