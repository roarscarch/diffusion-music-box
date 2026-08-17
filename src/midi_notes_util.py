import numpy as np


class MidiNotesUtil:
    """Utility functions for MIDI note conversion and frequency mapping.

    This module provides helper functions for converting MIDI note numbers to
    note names, frequencies, and frequency bin indices for spectrogram-based
    processing. It complements the MidiNotes class by adding pure functions
    that can be used standalone.
    """

    # MIDI note names (C, C#, D, ... B) for octaves -1 to 9
    NOTE_NAMES = [
        'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
    ]

    @staticmethod
    def note_number_to_name(midi_note):
        """Convert a MIDI note number to a note name (e.g., 60 -> 'C4').

        Parameters
        ----------
        midi_note : int
            MIDI note number (0-127).

        Returns
        -------
        str
            Note name with octave (e.g., 'C4', 'F#3').

        Raises
        ------
        ValueError
            If midi_note is outside 0-127.
        """
        if not 0 <= midi_note <= 127:
            raise ValueError(f"MIDI note must be between 0 and 127, got {midi_note}")
        pitch_class = midi_note % 12
        octave = midi_note // 12 - 1
        return f"{MidiNotesUtil.NOTE_NAMES[pitch_class]}{octave}"

    @staticmethod
    def note_name_to_number(note_name):
        """Convert a note name (e.g., 'C4') to a MIDI note number.

        Parameters
        ----------
        note_name : str
            Note name with optional accidental (#) and octave (e.g., 'C4', 'F#3').
            Octave defaults to 4 if not specified.

        Returns
        -------
        int
            MIDI note number (0-127).

        Raises
        ------
        ValueError
            If note_name is invalid or out of range.
        """
        note_name = note_name.strip().upper()
        if not note_name:
            raise ValueError("Note name cannot be empty")

        # Split into pitch class and octave
        # Find where digits start (octave)
        import re
        match = re.match(r'^([A-G]#?)(-?\d+)?$', note_name)
        if not match:
            raise ValueError(f"Invalid note name: {note_name}")
        pitch = match.group(1)
        octave_str = match.group(2)
        octave = int(octave_str) if octave_str else 4

        if pitch not in MidiNotesUtil.NOTE_NAMES:
            raise ValueError(f"Unknown pitch class: {pitch}")

        midi_note = MidiNotesUtil.NOTE_NAMES.index(pitch) + (octave + 1) * 12
        if not 0 <= midi_note <= 127:
            raise ValueError(f"MIDI note out of range: {midi_note}")
        return midi_note

    @staticmethod
    def note_number_to_frequency(midi_note):
        """Convert a MIDI note number to frequency in Hz.

        Parameters
        ----------
        midi_note : int
            MIDI note number (0-127).

        Returns
        -------
        float
            Frequency in Hz.
        """
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    @staticmethod
    def frequency_to_note_number(freq):
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
            raise ValueError(f"Frequency must be positive, got {freq}")
        return int(round(69 + 12 * np.log2(freq / 440.0)))

    @staticmethod
    def frequency_to_bin(freq, sample_rate, fft_size):
        """Convert a frequency in Hz to the nearest FFT bin index.

        Parameters
        ----------
        freq : float
            Frequency in Hz.
        sample_rate : int
            Sample rate of the audio.
        fft_size : int
            FFT size used for the spectrogram.

        Returns
        -------
        int
            Nearest FFT bin index.
        """
        if freq < 0:
            raise ValueError(f"Frequency must be non-negative, got {freq}")
        if fft_size <= 0:
            raise ValueError(f"FFT size must be positive, got {fft_size}")
        return int(round(freq * fft_size / sample_rate))

    @staticmethod
    def midi_notes_to_frequencies(midi_notes):
        """Convert a list of MIDI notes to frequencies.

        Parameters
        ----------
        midi_notes : list of int
            MIDI note numbers.

        Returns
        -------
        list of float
            Frequencies in Hz for each note.
        """
        return [MidiNotesUtil.note_number_to_frequency(n) for n in midi_notes]

    @staticmethod
    def midi_notes_to_bins(midi_notes, sample_rate, fft_size):
        """Convert a list of MIDI notes to FFT bin indices.

        Parameters
        ----------
        midi_notes : list of int
            MIDI note numbers.
        sample_rate : int
            Sample rate of the audio.
        fft_size : int
            FFT size used for the spectrogram.

        Returns
        -------
        list of int
            FFT bin indices for each note.
        """
        return [MidiNotesUtil.frequency_to_bin(
            MidiNotesUtil.note_number_to_frequency(n), sample_rate, fft_size
        ) for n in midi_notes]

    @staticmethod
    def get_note_range(start_note=21, end_note=108):
        """Get a list of MIDI note numbers in a range.

        Parameters
        ----------
        start_note : int, optional
            Starting MIDI note number (default 21 = A0).
        end_note : int, optional
            Ending MIDI note number (default 108 = C8).

        Returns
        -------
        list of int
            MIDI note numbers from start to end inclusive.

        Raises
        ------
        ValueError
            If start is greater than end or out of range.
        """
        if start_note > end_note:
            raise ValueError(f"start_note ({start_note}) must be <= end_note ({end_note})")
        if not 0 <= start_note <= 127 or not 0 <= end_note <= 127:
            raise ValueError("MIDI notes must be between 0 and 127")
        return list(range(start_note, end_note + 1))
