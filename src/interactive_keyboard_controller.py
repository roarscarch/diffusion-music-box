import numpy as np
import threading
import time
import sys


class InteractiveKeyboardController:
    """Interactive keyboard controller for real-time parameter adjustment.

    This module provides a simple keyboard-based interface for adjusting
    diffusion parameters in real-time. It runs in a background thread and
    listens for key presses on stdin, mapping them to parameter changes
    such as noise schedule, diffusion steps, and tempo. The controller
    maintains a set of current parameter values that can be read by other
    components, and it supports a callback mechanism for immediate updates.

    Parameters
    ----------
    initial_params : dict, optional
        Dictionary of initial parameter values. Keys can include:
        - 'noise_schedule': float in [0, 1] controlling noise schedule
        - 'diffusion_steps': int, number of diffusion steps
        - 'tempo': float, tempo in BPM
        - 'volume': float, output volume
        - 'crossfade': float, crossfade length in seconds
    """

    def __init__(self, initial_params=None):
        self.params = {
            'noise_schedule': 0.5,
            'diffusion_steps': 50,
            'tempo': 60.0,
            'volume': 0.8,
            'crossfade': 0.1,
        }
        if initial_params:
            self.params.update(initial_params)

        self._listener_thread = None
        self._running = False
        self._lock = threading.Lock()
        self._callbacks = []

    def start(self):
        """Start the keyboard listener in a background thread.

        The listener reads from stdin in a non-blocking manner and
        processes key presses. This method returns immediately.
        """
        if self._running:
            return
        self._running = True
        self._listener_thread = threading.Thread(target=self._listen, daemon=True)
        self._listener_thread.start()

    def stop(self):
        """Stop the keyboard listener thread."""
        self._running = False
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=1.0)
            self._listener_thread = None

    def add_callback(self, callback):
        """Register a callback to be invoked when parameters change.

        Parameters
        ----------
        callback : callable
            Function that accepts a dict of updated parameters.
        """
        with self._lock:
            self._callbacks.append(callback)

    def get_params(self):
        """Return a copy of the current parameters.

        Returns
        -------
        dict
            Current parameter values.
        """
        with self._lock:
            return dict(self.params)

    def _listen(self):
        """Background loop that reads character input and updates params."""
        import termios
        import tty
        import select

        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                # Check if there is input available
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch:
                        self._handle_key(ch)
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, ch):
        """Process a single key press.

        Mapping:
        - 'n' / 'N' : increase / decrease noise schedule
        - 'd' / 'D' : increase / decrease diffusion steps
        - 't' / 'T' : increase / decrease tempo
        - 'v' / 'V' : increase / decrease volume
        - 'c' / 'C' : increase / decrease crossfade
        - 'q' : quit (sets running to False)
        """
        updated = {}