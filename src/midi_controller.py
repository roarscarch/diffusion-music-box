import numpy as np
import threading
import time


class MidiController:
    """Handles MIDI input for interactive parameter control.

    This module provides a minimal MIDI controller interface that can be
    used to adjust parameters such as noise schedule, diffusion steps, and
    tempo in real time. It supports a simple callback-based API where
    MIDI messages are parsed and mapped to parameter changes. The
    implementation is backend-agnostic and can be wired to a MIDI library
    like mido or rtmidi in the future.

    Parameters
    ----------
    channel : int, optional
        MIDI channel to listen on (0-15). Default is 0.
    cc_map : dict, optional
        Mapping of MIDI control change numbers to parameter names.
        Default maps common CC numbers to relevant parameters.
    """

    def __init__(self, channel=0, cc_map=None):
        self.channel = channel
        self.cc_map = cc_map or {
            20: 'noise_schedule',
            21: 'diffusion_steps',
            22: 'tempo',
            23: 'volume',
        }
        self._listeners = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def add_listener(self, param, callback):
        """Register a callback for a specific parameter.

        Parameters
        ----------
        param : str
            Parameter name (e.g., 'tempo').
        callback : callable
            Function to call when the parameter changes. It receives
            the new value as a float in the range [0.0, 1.0] or an int
            for discrete params.
        """
        with self._lock:
            self._listeners.setdefault(param, []).append(callback)

    def _emit(self, param, value):
        """Emit a parameter change to all registered listeners."""
        with self._lock:
            callbacks = list(self._listeners.get(param, []))
        for cb in callbacks:
            try:
                cb(value)
            except Exception as e:
                print(f"MidiController: callback error for {param}: {e}")

    def handle_cc(self, cc_number, value):
        """Process a control change message.

        Parameters
        ----------
        cc_number : int
            MIDI control change number (0-127).
        value : int
            MIDI value (0-127).
        """
        if cc_number not in self.cc_map:
            return
        param = self.cc_map[cc_number]
        # Normalize to 0.0-1.0
        normalized = value / 127.0
        self._emit(param, normalized)

    def handle_note_on(self, note, velocity):
        """Process a note-on message for triggering actions.

        Parameters
        ----------
        note : int
            MIDI note number (0-127).
        velocity : int
            Note velocity (0-127).
        """
        # Map notes to simple commands, e.g., start/stop or preset selection
        if velocity == 0:
            return
        if note == 60:  # C4 - play/pause
            self._emit('toggle_play', 1.0)
        elif note == 61:  # C#4 - next preset
            self._emit('next_preset', 1.0)
        elif note == 62:  # D4 - previous preset
            self._emit('prev_preset', 1.0)

    def start(self):
        """Start the MIDI input thread (simulated for now).

        In a real implementation, this would open a MIDI port and listen
        for incoming messages. For now, it just marks the controller as
        running and prints a message.
        """
        self._running = True
        print("MidiController: started (simulated input)")

    def stop(self):
        """Stop the MIDI input thread."""
        self._running = False
        print("MidiController: stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
