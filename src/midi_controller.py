import mido
import threading
from typing import Callable, Dict, Optional


class MidiController:
    """Listens to MIDI control change messages and maps them to parameter callbacks.

    This class opens a MIDI input port and runs a background thread that
    processes incoming control change (CC) messages. Each CC number can be
    mapped to a callback function that receives the normalized value (0.0 to 1.0).

    Parameters
    ----------
    port_name : str, optional
        Name of the MIDI input port to open. If None, the default input port is used.
    """

    def __init__(self, port_name: Optional[str] = None):
        self.port_name = port_name
        self._callbacks: Dict[int, Callable[[float], None]] = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._port = None

    def open(self) -> bool:
        """Open the MIDI input port and start the listener thread.

        Returns
        -------
        bool
            True if the port was opened successfully, False otherwise.
        """
        try:
            if self.port_name is None:
                self._port = mido.open_input()
            else:
                self._port = mido.open_input(self.port_name)
        except (OSError, ValueError, IOError):
            return False
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        return True

    def close(self) -> None:
        """Stop the listener thread and close the MIDI port."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._port is not None:
            self._port.close()
            self._port = None

    def set_callback(self, cc_number: int, callback: Callable[[float], None]) -> None:
        """Register a callback for a specific control change number.

        Parameters
        ----------
        cc_number : int
            MIDI control change number (0-127).
        callback : Callable[[float], None]
            Function that takes a normalized value (0.0-1.0) as input.
        """
        self._callbacks[cc_number] = callback

    def _listen(self) -> None:
        """Background thread loop that processes incoming MIDI messages."""
        while self._running:
            try:
                msg = self._port.receive(block=True)
                if msg.type == 'control_change':
                    cc = msg.control
                    if cc in self._callbacks:
                        # Normalize value 0-127 to 0.0-1.0
                        normalized = msg.value / 127.0
                        self._callbacks[cc](normalized)
            except (OSError, ValueError, IOError):
                break

    @staticmethod
    def list_input_ports() -> list:
        """Return a list of available MIDI input ports."""
        try:
            return mido.get_input_names()
        except Exception:
            return []
