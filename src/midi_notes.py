import numpy as np


def midi_to_freq(midi_note):
    """Convert a MIDI note number to frequency in Hz.

    Uses the standard formula: 440 * 2^((note - 69) / 12).

    Parameters
    ----------
    midi_note : int or float
        MIDI note number (0-127).

    Returns
    -------
    float
        Frequency in Hz.

    Raises
    ------
    ValueError
        If the note number is outside the valid MIDI range.
    """
    note = float(midi_note)
    if note < 0 or note > 127:
        raise ValueError(f"MIDI note must be between 0 and 127, got {midi_note}")
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def freq_to_midi(freq):
    """Convert a frequency in Hz to the nearest MIDI note number.

    Parameters
    ----------
    freq : float
        Frequency in Hz. Must be positive.

    Returns
    -------
    int
        Closest MIDI note number.

    Raises
    ------
    ValueError
        If frequency is not positive.
    """
    freq = float(freq)
    if freq <= 0:
        raise ValueError(f"Frequency must be positive, got {freq}")
    midi = 69 + 12 * np.log2(freq / 440.0)
    return int(round(midi))


def midi_to_name(midi_note):
    """Convert a MIDI note number to a human-readable note name.

    Parameters
    ----------
    midi_note : int
        MIDI note number (0-127).

    Returns
    -------
    str
        Note name like 'C4', 'F#5', etc.

    Raises
    ------
    ValueError
        If the note number is outside the valid MIDI range.
    """
    note = int(midi_note)
    if note < 0 or note > 127:
        raise ValueError(f"MIDI note must be between 0 and 127, got {midi_note}")
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = note // 12 - 1
    name = names[note % 12]
    return f"{name}{octave}"
