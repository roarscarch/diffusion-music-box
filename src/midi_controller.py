import numpy as np
import threading
import time


class MIDIController:
    """Keyboard-based control for interactive parameters.

    This controller listens for keyboard input in a background thread and
    maps keys to parameter adjustments. It is designed to be used in a
    headless CLI environment, providing a MIDI-like experience without
    requiring actual MIDI hardware.

    Parameters
    ----------
    param_holder : object
        An object with attributes that can be adjusted (e.g., noise_schedule,
        diffusion_steps, tempo). The controller will modify these in place.
    """

    def __init__(self, param_holder):
        self.param_holder = param_holder
        self._running = False
        self._thread = None
        self._keymap = {
            'q': ('tempo', -5.0),
            'w': ('tempo', 5.0),
            'a': ('diffusion_steps', -1),
            's': ('diffusion_steps', 1),
            'z': ('noise_schedule', 'prev'),
            'x': ('noise_schedule', 'next'),
        }
        self._schedule_cycle = ['linear', 'cosine', 'quadratic']

    def start(self):
        """Start the input listener thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the input listener thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _listen(self):
        """Background loop reading keyboard input."""
        try:
            import tty
            import termios
            import sys

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self._running:
                    ch = sys.stdin.read(1)
                    if not ch:
                        continue
                    if ch == '\x1b':  # ESC
                        break
                    self._handle_key(ch.lower())
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except ImportError:
            # Fallback for non-POSIX systems: read lines
            while self._running:
                try:
                    line = input()
                    if not line:
                        continue
                    self._handle_key(line.strip().lower())
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break

    def _handle_key(self, key):
        """Apply the key mapping to adjust parameters."""
        if key not in self._keymap:
            return
        param, delta = self._keymap[key]
        if param == 'noise_schedule':
            self._cycle_schedule(delta)
        else:
            current = getattr(self.param_holder, param, None)
            if current is not None:
                setattr(self.param_holder, param, current + delta)
                print(f"{param} = {getattr(self.param_holder, param)}")

    def _cycle_schedule(self, direction):
        """Cycle through noise schedules."""
        current = getattr(self.param_holder, 'noise_schedule', 'linear')
        try:
            idx = self._schedule_cycle.index(current)
        except ValueError:
            idx = 0
        if direction == 'next':
            idx = (idx + 1) % len(self._schedule_cycle)
        else:
            idx = (idx - 1) % len(self._schedule_cycle)
        new_schedule = self._schedule_cycle[idx]
        setattr(self.param_holder, 'noise_schedule', new_schedule)
        print(f"noise_schedule = {new_schedule}")
