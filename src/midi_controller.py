import numpy as np
import threading
import time
from collections import deque


class MIDIKeyboardController:
    """Headless MIDI-like control via keyboard for interactive parameter adjustment.

    The controller listens for keyboard input in a background thread and maps
    key presses to parameter changes. It maintains a thread-safe parameter
    dictionary that can be read by the main application loop.

    Supported keys:
        - 'n' / 'N' : increase / decrease diffusion steps
        - 's' / 'S' : increase / decrease noise scale (0.0 to 1.0)
        - 't' / 'T' : increase / decrease tempo (BPM)
        - ' ' (space) : toggle pause / resume generation
        - 'q' : quit
    """

    def __init__(self, initial_params=None):
        self._params = {
            'diffusion_steps': 50,
            'noise_scale': 0.6,
            'tempo': 120.0,
            'paused': False,
            'quit': False,
        }
        if initial_params:
            self._params.update(initial_params)
        self._lock = threading.Lock()
        self._key_queue = deque()
        self._listener_thread = None
        self._running = False

    def start(self):
        """Start the keyboard listener in a background thread."""
        if self._running:
            return
        self._running = True
        self._listener_thread = threading.Thread(target=self._listen, daemon=True)
        self._listener_thread.start()

    def stop(self):
        """Stop the keyboard listener."""
        self._running = False
        if self._listener_thread:
            self._listener_thread.join(timeout=1.0)
            self._listener_thread = None

    def _listen(self):
        """Background thread that reads keyboard input."""
        import sys
        import termios
        import tty

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self._running:
                ch = sys.stdin.read(1)
                if ch:
                    with self._lock:
                        self._key_queue.append(ch)
        except Exception:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def poll_events(self):
        """Process any pending key events and update parameters.

        Returns a list of strings describing the changes made.
        """
        changes = []
        with self._lock:
            while self._key_queue:
                key = self._key_queue.popleft()
                change = self._handle_key(key)
                if change:
                    changes.append(change)
        return changes

    def _handle_key(self, key):
        """Handle a single key press and update parameters.

        Returns a description string if a parameter changed, else None.
        """
        if key == 'n':
            self._params['diffusion_steps'] = min(200, self._params['diffusion_steps'] + 10)
            return f"diffusion_steps -> {self._params['diffusion_steps']}"
        elif key == 'N':
            self._params['diffusion_steps'] = max(1, self._params['diffusion_steps'] - 10)
            return f"diffusion_steps -> {self._params['diffusion_steps']}"
        elif key == 's':
            self._params['noise_scale'] = min(1.0, self._params['noise_scale'] + 0.05)
            return f"noise_scale -> {self._params['noise_scale']:.2f}"
        elif key == 'S':
            self._params['noise_scale'] = max(0.0, self._params['noise_scale'] - 0.05)
            return f"noise_scale -> {self._params['noise_scale']:.2f}"
        elif key == 't':
            self._params['tempo'] = min(240.0, self._params['tempo'] + 5.0)
            return f"tempo -> {self._params['tempo']:.1f}"
        elif key == 'T':
            self._params['tempo'] = max(40.0, self._params['tempo'] - 5.0)
            return f"tempo -> {self._params['tempo']:.1f}"
        elif key == ' ':
            self._params['paused'] = not self._params['paused']
            state = "paused" if self._params['paused'] else "resumed"
            return f"{state}"
        elif key == 'q':
            self._params['quit'] = True
            return "quit requested"
        return None

    def get_params(self):
        """Return a copy of the current parameters."""
        with self._lock:
            return dict(self._params)

    def set_param(self, name, value):
        """Set a parameter by name."""
        with self._lock:
            self._params[name] = value
