import numpy as np
import threading
import time


class MidiController:
    """A keyboard-based controller for interacting with the diffusion music box.

    This module provides a simple interface to map keyboard keys to parameter
    changes in the generation pipeline. It listens for key presses and adjusts
    parameters such as noise schedule, diffusion steps, and tempo. The controller
    can run in a background thread and supports both interactive and headless
    usage.

    Parameters
    ----------
    sample_rate : int, optional
        Sample rate of the audio engine (unused but kept for interface consistency).
    """

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._callbacks = {}
        self._recording = False
        self._recorded_events = []

    def register_callback(self, key, callback):
        """Register a callback for a specific key.

        Parameters
        ----------
        key : str
            The key to listen for (e.g., 'a', 'left', 'space').
        callback : callable
            Function to call when the key is pressed. It receives a single
            argument: the key name as a string.
        """
        with self._lock:
            self._callbacks[key.lower()] = callback

    def start(self):
        """Start the controller in a background thread.

        This method spawns a thread that reads keyboard input. It requires a
        terminal that supports raw input (e.g., on Linux/macOS with termios).
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the controller thread gracefully."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self):
        """Main loop for reading keyboard input.

        Uses termios to set raw mode on Unix-like systems. Falls back to
        reading from sys.stdin with a simple line-based input if raw mode
        is not available (e.g., on Windows).
        """
        try:
            import termios
            import sys
            import tty

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
        except (ImportError, AttributeError):
            # Fallback for Windows or non-terminal environments
            import sys
            while self._running:
                line = sys.stdin.readline().strip()
                if not line:
                    continue
                for ch in line:
                    self._handle_key(ch)

    def _handle_key(self, key):
        """Process a single key press.

        Parameters
        ----------
        key : str
            The key character or escape sequence name.
        """
        key = key.lower()
        if key == '\x1b':
            # Escape sequences for arrow keys are multi-byte; we ignore them here
            return
        with self._lock:
            callback = self._callbacks.get(key)
        if callback:
            try:
                callback(key)
            except Exception as e:
                print(f"Error in callback for key '{key}': {e}")
        if self._recording:
            self._recorded_events.append((time.time(), key))

    def start_recording(self):
        """Start recording key press events."""
        self._recording = True
        self._recorded_events = []

    def stop_recording(self):
        """Stop recording and return the recorded events.

        Returns
        -------
        list of tuple
            List of (timestamp, key) tuples recorded since start_recording.
        """
        self._recording = False
        return self._recorded_events
