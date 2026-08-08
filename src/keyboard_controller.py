import sys
import termios
import tty
import threading


class KeyboardController:
    """Reads single keystrokes in a separate thread and maps them to parameter changes.

    This controller provides a simple, headless way to adjust parameters in real time
    without requiring MIDI hardware. It runs a background thread that reads from stdin
    in raw mode and invokes a user-supplied callback for each keypress.

    Parameters
    ----------
    callback : callable
        Function called with a single character (string) for each keypress.
    """

    def __init__(self, callback):
        self._callback = callback
        self._stop_event = threading.Event()
        self._thread = None
        self._fd = None
        self._old_settings = None

    def start(self):
        """Start the keyboard listener in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the keyboard listener and restore terminal settings."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._fd is not None and self._old_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def _run(self):
        try:
            tty.setraw(self._fd)
            while not self._stop_event.is_set():
                ch = sys.stdin.read(1)
                if ch:
                    self._callback(ch)
        except Exception:
            pass
        finally:
            if self._fd is not None and self._old_settings is not None:
                try:
                    termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
                except Exception:
                    pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
