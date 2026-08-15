import numpy as np


# MIDI note number to note name mapping (C, C#, D, ...)
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_freq(midi_note):
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


def freq_to_midi(freq):
    """Convert a frequency in Hz to the nearest MIDI note number.

    Parameters
    ----------
    freq : float
        Frequency in Hz.

    Returns
    -------
    int
        MIDI note number (0-127).
    """
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_note_name(midi_note):
    """Convert a MIDI note number to a note name string with octave.

    Parameters
    ----------
    midi_note : int
        MIDI note number (0-127).

    Returns
    -------
    str
        Note name, e.g., 'C4', 'F#3'.
    """
    if midi_note < 0 or midi_note > 127:
        raise ValueError(f"MIDI note out of range: {midi_note}")
    note_name = _NOTE_NAMES[midi_note % 12]
    octave = midi_note // 12 - 1
    return f"{note_name}{octave}"


def note_name_to_midi(note_name):
    """Convert a note name string (e.g., 'C4') to MIDI note number.

    Parameters
    ----------
    note_name : str
        Note name in format like 'C4', 'F#3', 'Bb2' (case-insensitive).

    Returns
    -------
    int
        MIDI note number.
    """
    note_name = note_name.strip().upper()
    if not note_name:
        raise ValueError("Empty note name")
    # Find the letter part (C, D, E, F, G, A, B) and optional sharp/flat
    letter = note_name[0]
    if letter not in 'ABCDEFG':
        raise ValueError(f"Invalid note letter: {letter}")
    idx = 1
    accidental = 0
    if idx < len(note_name) and note_name[idx] == '#':
        accidental = 1
        idx += 1
    elif idx < len(note_name) and note_name[idx] == 'B':
        accidental = -1
        idx += 1
    # Parse octave (remaining characters)
    if idx >= len(note_name):
        raise ValueError(f"Missing octave in note name: {note_name}")
    try:
        octave = int(note_name[idx:])
    except ValueError:
        raise ValueError(f"Invalid octave in note name: {note_name}")
    # Map letter to base pitch class
    base = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[letter]
    pitch_class = (base + accidental) % 12
    midi_note = (octave + 1) * 12 + pitch_class
    if midi_note < 0 or midi_note > 127:
        raise ValueError(f"MIDI note out of range: {midi_note}")
    return midi_note


def midi_to_freq_array(midi_notes):
    """Convert an array of MIDI notes to frequencies.

    Parameters
    ----------
    midi_notes : array-like
        Sequence of MIDI note numbers.

    Returns
    -------
    np.ndarray
        Array of frequencies in Hz.
    """
    return np.array([midi_to_freq(n) for n in midi_notes], dtype=np.float32)


def freq_to_midi_array(freqs):
    """Convert an array of frequencies to nearest MIDI notes.

    Parameters
    ----------
    freqs : array-like
        Sequence of frequencies in Hz.

    Returns
    -------
    np.ndarray
        Array of MIDI note numbers.
    """
    return np.array([freq_to_midi(f) for f in freqs], dtype=int)
