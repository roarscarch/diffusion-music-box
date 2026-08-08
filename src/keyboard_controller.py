import numpy as np
import threading
import time
from enum import Enum


class ControlParam(Enum):
    NOISE_SCHEDULE = 'noise_schedule'
    DIFFUSION_STEPS = 'diffusion_steps'
    TEMPO = 'tempo'
    GAIN = 'gain'


class KeyboardController:
    """Provide real-time parameter control via keyboard input.

    This class listens for keyboard events in a background thread and maps
    key presses to parameter adjustments. It is designed to be used in a
    headless environment where the user can tweak generation parameters
    without a GUI.

    The mapping is as follows:
    - 'n' : increase noise schedule (more noise)
    - 'N' : decrease noise schedule (less noise)
    - 's' : increase diffusion steps
    - 'S' : decrease diffusion steps
    - 't' : increase tempo (BPM)
    - 'T' : decrease tempo (BPM)
    - 'g' : increase gain
    - 'G' : decrease gain
    - 'q' : quit (sets a flag)

    Parameters
    ----------
    initial_params : dict, optional
        Initial values for the controllable parameters. Keys should be
        members of :class:`ControlParam`. If omitted, sensible defaults are
        used.
    """

    def __init__(self, initial_params=None):
        self.params = {
            ControlParam.NOISE_SCHEDULE: 0.5,
            ControlParam.DIFFUSION_STEPS: 50,
            ControlParam.TEMPO: 60.0,
            ControlParam.GAIN: 0.5,
        }
        if initial_params:
            for key, value in initial_params.items():
                if key in self.params:
                    self.params[key] = float(value)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._quit_flag = False

    def start(self):
        """Start the keyboard listener thread."""
        if self._running:
            return
        self._running = True
        self._quit_flag = False
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the keyboard listener thread."""
        self._running = False
        self._quit_flag = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _listen(self):
        """Background loop that reads keyboard input."""
        while self._running:
            try:
                # In a real terminal, this would use something like `keyboard`
                # or `pynput`. For now, we provide a dummy that reads from stdin
                # with a timeout to keep the loop responsive.
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline()
                    if line:
                        self._handle_key(line.strip())
                else:
                    time.sleep(0.05)
            except KeyboardInterrupt:
                self._quit_flag = True
                break

    def _handle_key(self, key):
        """Process a single key press."""
        if not key:
            return
        # Support both single-character keys and longer strings (e.g., 'N')
        # We only care about the first character for simplicity.
        k = key[0]
        with self._lock:
            if k == 'q':
                self._quit_flag = True
            elif k == 'n':
                self.params[ControlParam.NOISE_SCHEDULE] = min(1.0, self.params[ControlParam.NOISE_SCHEDULE] + 0.05)
            elif k == 'N':
                self.params[ControlParam.NOISE_SCHEDULE] = max(0.0, self.params[ControlParam.NOISE_SCHEDULE] - 0.05)
            elif k == 's':
                self.params[ControlParam.DIFFUSION_STEPS] = min(200, int(self.params[ControlParam.DIFFUSION_STEPS]) + 5)
            elif k == 'S':
                self.params[ControlParam.DIFFUSION_STEPS] = max(1, int(self.params[ControlParam.DIFFUSION_STEPS]) - 5)
            elif k == 't':
                self.params[ControlParam.TEMPO] = min(200.0, self.params[ControlParam.TEMPO] + 2.0)
            elif k == 'T':
                self.params[ControlParam.TEMPO] = max(20.0, self.params[ControlParam.TEMPO] - 2.0)
            elif k == 'g':
                self.params[ControlParam.GAIN] = min(1.0, self.params[ControlParam.GAIN] + 0.05)
            elif k == 'G':
                self.params[ControlParam.GAIN] = max(0.0, self.params[ControlParam.GAIN] - 0.05)

    def get_param(self, param):
        """Retrieve the current value of a parameter.

        Parameters
        ----------
        param : ControlParam
            The parameter to query.

        Returns
        -------
        float or int
            The current value. Integer values are converted to int for
            parameters that are inherently integer-based.
        """
        with self._lock:
            val = self.params[param]
            if param in (ControlParam.DIFFUSION_STEPS,):
                return int(val)
            return val

    def set_param(self, param, value):
        """Set a parameter to a specific value.

        Parameters
        ----------
        param : ControlParam
            The parameter to set.
        value : float or int
            The new value.
        """
        with self._lock:
            self.params[param] = float(value)

    def quit_requested(self):
        """Return True if the user requested to quit."""
        return self._quit_flag

    def get_all_params(self):
        """Return a copy of all current parameters as a dict."""
        with self._lock:
            return {k: v for k, v in self.params.items()}