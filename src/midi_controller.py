import numpy as np
import threading
import time


class MidiController:
    """Headless MIDI-like control via keyboard.

    This controller listens for keyboard input in a background thread and
    translates key presses into control signals for the music generation
    pipeline. It supports adjusting the noise schedule, diffusion steps,
    tempo, and triggering regeneration of the current segment.

    Parameters
    ----------
    update_callback : callable, optional
        A function that receives a dictionary of parameter updates.
        The keys are parameter names and values are the new values.
    """

    def __init__(self, update_callback=None):
        self.update_callback = update_callback
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._parameters = {
            'noise_schedule': 'linear',
            'diffusion_steps': 50,
            'tempo': 120.0,
        }

    def start(self):
        """Start the keyboard listener in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the keyboard listener."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _listen(self):
        """Background thread that reads keyboard input."""
        import sys
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                self._handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key):
        """Handle a single key press."""
        with self._lock:
            params = {}
            if key == 'n':
                # Cycle noise schedule
                current = self._parameters['noise_schedule']
                schedules = ['linear', 'cosine']
                idx = schedules.index(current) if current in schedules else 0
                new = schedules[(idx + 1) % len(schedules)]
                self._parameters['noise_schedule'] = new
                params['noise_schedule'] = new
            elif key == 'd':
                # Increase diffusion steps
                steps = min(200, self._parameters['diffusion_steps'] + 10)
                self._parameters['diffusion_steps'] = steps
                params['diffusion_steps'] = steps
            elif key == 'a':
                # Decrease diffusion steps
                steps = max(10, self._parameters['diffusion_steps'] - 10)
                self._parameters['diffusion_steps'] = steps
                params['diffusion_steps'] = steps
            elif key == 't':
                # Increase tempo
                tempo = min(240.0, self._parameters['tempo'] + 5.0)
                self._parameters['tempo'] = tempo
                params['tempo'] = tempo
            elif key == 'g':
                # Decrease tempo
                tempo = max(40.0, self._parameters['tempo'] - 5.0)
                self._parameters['tempo'] = tempo
                params['tempo'] = tempo
            elif key == 'r':
                # Regenerate current segment
                params['regenerate'] = True

            if params and self.update_callback:
                self.update_callback(params)

    def get_parameters(self):
        """Return a copy of the current parameters."""
        with self._lock:
            return dict(self._parameters)
