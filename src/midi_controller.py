import numpy as np
import mido


class MIDIController:
    """Handles MIDI input for real-time parameter control.

    This controller listens to MIDI messages on a selected input port and
    translates them into parameter updates for the generative music system.
    It supports control change (CC) messages for continuous parameters and
    note on/off messages for discrete actions (e.g., triggering a new tile).
    """

    def __init__(self, port_name=None, callback=None):
        self.port_name = port_name
        self.callback = callback
        self._port = None
        self._running = False

    def open(self):
        """Open the MIDI input port.

        Returns
        -------
        bool
            True if the port opened successfully, False otherwise.
        """
        try:
            if self.port_name is None:
                # Use the default input port
                self._port = mido.open_input()
            else:
                self._port = mido.open_input(self.port_name)
            self._running = True
            return True
        except Exception as e:
            print(f"Failed to open MIDI port: {e}")
            return False

    def close(self):
        """Close the MIDI input port."""
        if self._port is not None:
            self._port.close()
            self._port = None
        self._running = False

    def poll(self):
        """Poll for incoming MIDI messages and dispatch them.

        Call this method regularly from the main loop to process MIDI input.
        """
        if not self._running or self._port is None:
            return
        for msg in self._port.iter_pending():
            self._handle_message(msg)

    def _handle_message(self, msg):
        """Process a single MIDI message.

        Parameters
        ----------
        msg : mido.Message
            The incoming MIDI message.
        """
        if msg.type == 'control_change':
            # Map CC number to parameter name
            param_map = {
                1: 'noise_schedule',
                2: 'diffusion_steps',
                3: 'tempo',
            }