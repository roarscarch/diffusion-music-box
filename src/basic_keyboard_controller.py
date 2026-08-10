import numpy as np
import threading


class BasicKeyboardController:
    """A simple keyboard controller for interactive parameter adjustment.

    This module provides a thread-safe way to adjust parameters such as
    diffusion steps, noise schedule, and tempo based on keyboard input.
    It listens for key presses in a background thread and updates
    internal state that can be polled by the main loop.

    Parameters
    ----------
    initial_params : dict, optional
        Initial values for parameters. Keys can include 'diffusion_steps',
        'noise_schedule', 'tempo'.
    """

    def __init__(self, initial_params=None):
        self.params = {
            'diffusion_steps': 10,
            'noise_schedule': 'linear',
            'tempo': 60,
        }
        if initial_params:
            self.params.update(initial_params)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """Start the keyboard listener thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the keyboard listener thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _listen(self):
        """Background thread that reads keyboard input."""
        import sys
        import termios
        import tty
        import select

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch:
                        self._handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key):
        """Process a single key press and update parameters."""
        with self._lock:
            if key == 'w':
                self.params['diffusion_steps'] = min(
                    self.params['diffusion_steps'] + 1, 50)
            elif key == 's':
                self.params['diffusion_steps'] = max(
                    self.params['diffusion_steps'] - 1, 1)
            elif key == 'a':
                self.params['tempo'] = max(
                    self.params['tempo'] - 1, 30)
            elif key == 'd':
                self.params['tempo'] = min(
                    self.params['tempo'] + 1, 180)
            elif key == 'n':
                schedules = ['linear', 'cosine', 'exponential']
                idx = schedules.index(self.params['noise_schedule'])
                self.params['noise_schedule'] = schedules[(idx + 1) % len(schedules)]

    def get_params(self):
        """Return a copy of the current parameters.

        Returns
        -------
        dict
            A dictionary with keys 'diffusion_steps', 'noise_schedule', and 'tempo'.
        """
        with self._lock:
            return dict(self.params)
