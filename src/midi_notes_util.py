import numpy as np


def midi_to_freq(midi_note: int) -> float:
    """Convert a MIDI note number to frequency in Hz.

    Uses the standard formula: 440 * 2^((midi - 69) / 12).

    Parameters
    ----------
    midi_note : int
        MIDI note number (0-127).

    Returns
    -------
    float
        Frequency in Hz.

    Raises
    ------
    ValueError
        If midi_note is outside the valid range.
    """
    if not 0 <= midi_note <= 127:
        raise ValueError(f"MIDI note must be between 0 and 127, got {midi_note}")
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def freq_to_midi(freq: float) -> float:
    """Convert a frequency in Hz to the nearest MIDI note number (float).

    Parameters
    ----------
    freq : float
        Frequency in Hz. Must be positive.

    Returns
    -------
    float
        MIDI note number as a float (not rounded).

    Raises
    ------
    ValueError
        If freq is not positive.
    """
    if freq <= 0:
        raise ValueError(f"Frequency must be positive, got {freq}")
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_name(midi_note: int) -> str:
    """Return the note name (e.g., 'C4') for a MIDI note number.

    Parameters
    ----------
    midi_note : int
        MIDI note number (0-127).

    Returns
    -------
    str
        Note name with octave, e.g., 'C4' or 'F#3'.

    Raises
    ------
    ValueError
        If midi_note is outside the valid range.
    """
    if not 0 <= midi_note <= 127:
        raise ValueError(f"MIDI note must be between 0 and 127, got {midi_note}")
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    name = note_names[midi_note % 12]
    octave = midi_note // 12 - 1
    return f"{name}{octave}"


def name_to_midi(name: str) -> int:
    """Convert a note name (e.g., 'C4') to a MIDI note number.

    Supports sharps (#) and flats (b). Octave is required.

    Parameters
    ----------
    name : str
        Note name like 'C4', 'F#3', 'Bb2'.

    Returns
    -------
    int
        MIDI note number.

    Raises
    ------
    ValueError
        If the name is invalid.
    """
    name = name.strip()
    if not name:
        raise ValueError("Note name cannot be empty")

    # Parse accidental
    if len(name) >= 2 and name[1] in ('#', 'b'):
        note_part = name[0:2]
        octave_part = name[2:]
    else:
        note_part = name[0]
        octave_part = name[1:]

    note_names = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }

    if note_part not in note_names:
        raise ValueError(f"Unknown note: {note_part}")

    if not octave_part:
        raise ValueError("Octave is required (e.g., 'C4')")

    try:
        octave = int(octave_part)
    except ValueError:
        raise ValueError(f"Invalid octave: {octave_part}")

    midi = note_names[note_part] + (octave + 1) * 12
    if not 0 <= midi <= 127:
        raise ValueError(f"MIDI note out of range: {midi}")
    return midi
