import numpy as np


def midi_to_frequency(midi_note):
    """Convert a MIDI note number to its frequency in Hz.

    Parameters
    ----------
    midi_note : int or float
        MIDI note number (e.g., 69 for A4).

    Returns
    -------
    float
        Frequency in Hz.
    """
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def frequency_to_midi(frequency):
    """Convert a frequency in Hz to the nearest MIDI note number.

    Parameters
    ----------
    frequency : float
        Frequency in Hz.

    Returns
    -------
    int
        MIDI note number.
    """
    return int(round(69 + 12 * np.log2(frequency / 440.0)))


def midi_to_name(midi_note):
    """Convert a MIDI note number to a human-readable note name.

    Parameters
    ----------
    midi_note : int
        MIDI note number.

    Returns
    -------
    str
        Note name like 'C4', 'F#3'.
    """
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    name = names[midi_note % 12]
    octave = midi_note // 12 - 1
    return f"{name}{octave}"


def name_to_midi(note_name):
    """Convert a note name to MIDI note number.

    Parameters
    ----------
    note_name : str
        Note name like 'C4', 'F#3'.

    Returns
    -------
    int
        MIDI note number.

    Raises
    ------
    ValueError
        If the note name is invalid.
    """
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = note_name.strip()
    if len(note_name) < 2:
        raise ValueError(f"Invalid note name: {note_name}")
    # Separate note part and octave part
    if note_name[1] == '#':
        note_part = note_name[:2]
        octave_part = note_name[2:]
    else:
        note_part = note_name[:1]
        octave_part = note_name[1:]
    if note_part not in names:
        raise ValueError(f"Invalid note name: {note_name}")
    if not octave_part.isdigit():
        raise ValueError(f"Invalid octave: {note_name}")
    octave = int(octave_part)
    return (octave + 1) * 12 + names.index(note_part)


def midi_to_frequency_array(midi_notes):
    """Convert an array of MIDI notes to frequencies.

    Parameters
    ----------
    midi_notes : array-like
        MIDI note numbers.

    Returns
    -------
    np.ndarray
        Frequencies in Hz.
    """
    return midi_to_frequency(np.asarray(midi_notes))


def frequency_to_midi_array(frequencies):
    """Convert an array of frequencies to MIDI notes.

    Parameters
    ----------
    frequencies : array-like
        Frequencies in Hz.

    Returns
    -------
    np.ndarray
        MIDI note numbers.
    """
    return frequency_to_midi(np.asarray(frequencies))
