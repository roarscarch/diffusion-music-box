import numpy as np


class MIDINotes:
    """Convert between MIDI note numbers, names, and frequencies.

    This utility class provides static methods to translate MIDI note numbers
    to human-readable names (e.g., C4, F#5) and frequencies in Hz. It is used
    by the arpeggiator and keyboard controllers to interpret musical input.
    """

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    @staticmethod
    def midi_to_name(midi_note):
        """Convert a MIDI note number to a note name like 'C4'.

        Parameters
        ----------
        midi_note : int
            MIDI note number (0-127).

        Returns
        -------
        str
            Note name in scientific pitch notation (e.g., 'C4', 'F#3').

        Raises
        ------
        ValueError
            If midi_note is outside the valid range.
        """
        if not 0 <= midi_note <= 127:
            raise ValueError(f"MIDI note must be between 0 and 127, got {midi_note}")
        octave = midi_note // 12 - 1
        note_idx = midi_note % 12
        return f"{MIDINotes.NOTE_NAMES[note_idx]}{octave}"

    @staticmethod
    def name_to_midi(name):
        """Convert a note name to a MIDI note number.

        Parameters
        ----------
        name : str
            Note name in scientific pitch notation (e.g., 'C4', 'F#3').

        Returns
        -------
        int
            MIDI note number.

        Raises
        ------
        ValueError
            If the name is not valid.
        """
        name = name.strip()
        if not name:
            raise ValueError("Note name cannot be empty")
        # Separate pitch class and octave
        idx = 0
        while idx < len(name) and name[idx].isalpha():
            idx += 1
        if idx == 0 or idx == len(name):
            raise ValueError(f"Invalid note name: {name}")
        pitch_str = name[:idx]
        octave_str = name[idx:]
        try:
            octave = int(octave_str)
        except ValueError:
            raise ValueError(f"Invalid octave in note name: {name}")
        # Find pitch class index
        try:
            note_idx = MIDINotes.NOTE_NAMES.index(pitch_str.upper())
        except ValueError:
            raise ValueError(f"Unknown note name: {pitch_str}")
        return (octave + 1) * 12 + note_idx

    @staticmethod
    def midi_to_freq(midi_note):
        """Convert a MIDI note number to frequency in Hz.

        Parameters
        ----------
        midi_note : int
            MIDI note number.

        Returns
        -------
        float
            Frequency in Hz.
        """
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    @staticmethod
    def freq_to_midi(freq):
        """Convert a frequency in Hz to the nearest MIDI note number.

        Parameters
        ----------
        freq : float
            Frequency in Hz.

        Returns
        -------
        int
            Nearest MIDI note number.
        """
        if freq <= 0:
            raise ValueError("Frequency must be positive")
        return int(round(69 + 12 * np.log2(freq / 440.0)))

    @staticmethod
    def freq_to_name(freq):
        """Convert a frequency in Hz to the nearest note name.

        Parameters
        ----------
        freq : float
            Frequency in Hz.

        Returns
        -------
        str
            Note name (e.g., 'A4').
        """
        midi = MIDINotes.freq_to_midi(freq)
        return MIDINotes.midi_to_name(midi)
