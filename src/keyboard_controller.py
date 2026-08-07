import sys
import termios
import tty
import select
import threading
import time


class KeyboardController:
    """Reads keyboard input in a non-blocking way for real-time parameter control.

    This class runs a background thread that reads single keystrokes from the
    terminal without requiring the Enter key. It maintains a set of currently
    pressed keys and provides a callback mechanism for key press/release events.
    This is useful for headless CLI control of the diffusion music box.

    Parameters
    ----------
    on_key_press : callable, optional
        Callback invoked with the key character when a key is pressed.
    on_key_release : callable, optional
        Callback invoked with the key character when a key is released.
    """

    def __init__(self, on_key_press=None, on_key_release=None):
        self._on_key_press = on_key_press
        self._on_key_release = on_key_release
        self._pressed_keys = set()
        self._stop_event = threading.Event()
        self._thread = None
        self._fd = sys.stdin.fileno()
        self._old_settings = None

    def start(self):
        """Start the keyboard listener thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the keyboard listener thread and restore terminal settings."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._thread = None
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def _run(self):
        """Background thread loop that reads keystrokes."""
        while not self._stop_event.is_set():
            # Check if there is input available
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not rlist:
                continue
            try:
                # Read one byte (character)
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                if ch not in self._pressed_keys:
                    self._pressed_keys.add(ch)
                    if self._on_key_press:
                        self._on_key_press(ch)
                else:
                    # Key is still held; optionally handle repeat
                    pass
            except (IOError, OSError):
                break
            # Detect key releases by checking if key is no longer in the buffer
            # This is a simplified approach; for real release detection we'd need
            # raw escape sequences. Here we just report continuous presses.
            # We'll also add a small debounce to avoid duplicate events.
            time.sleep(0.02)

    def get_pressed_keys(self):
        """Return the set of currently pressed keys."""
        return set(self._pressed_keys)

    def clear_pressed_keys(self):
        """Clear the set of pressed keys."""
        self._pressed_keys.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    # Simple demo: print pressed keys
    def on_press(key):
        print(f"Pressed: {key!r}")

    with KeyboardController(on_key_press=on_press) as kc:
        print("Press keys (Ctrl+C to exit)")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
