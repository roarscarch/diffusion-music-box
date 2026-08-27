import numpy as np


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

    Examples
    --------
    >>> midi_to_freq(69)
    440.0
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
        Nearest MIDI note number.

    Examples
    --------
    >>> freq_to_midi(440.0)
    69
    """
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_bin(midi_note, sample_rate, fft_size):
    """Convert a MIDI note number to the nearest FFT bin index.

    Parameters
    ----------
    midi_note : int
        MIDI note number.
    sample_rate : int
        Sample rate of the audio.
    fft_size : int
        FFT size (number of frequency bins).

    Returns
    -------
    int
        FFT bin index (0 to fft_size//2).

    Examples
    --------
    >>> midi_to_bin(69, 22050, 1024)
    20
    """
    freq = midi_to_freq(midi_note)
    return int(round(freq * fft_size / sample_rate))


def bin_to_midi(bin_idx, sample_rate, fft_size):
    """Convert an FFT bin index to the nearest MIDI note number.

    Parameters
    ----------
    bin_idx : int
        FFT bin index (0 to fft_size//2).
    sample_rate : int
        Sample rate of the audio.
    fft_size : int
        FFT size (number of frequency bins).

    Returns
    -------
    int
        Nearest MIDI note number.

    Examples
    --------
    >>> bin_to_midi(20, 22050, 1024)
    69
    """
    freq = bin_idx * sample_rate / fft_size
    return freq_to_midi(freq)
