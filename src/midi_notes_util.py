import numpy as np


class MidiNotesUtil:
    """Utility class for converting between MIDI note numbers, frequencies, and note names.

    This module provides helpers for musical pitch conversions used throughout the
    project, such as mapping MIDI notes to frequencies for the arpeggiator or
    spectrogram frequency bins. It also includes a small lookup table for note names.
    """

    # Standard note names for MIDI note numbers (C4 = 60, A4 = 69)
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    @staticmethod
    def note_to_freq(midi_note: int) -> float:
        """Convert a MIDI note number to frequency in Hz.

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
            If midi_note is outside the valid range [0, 127].
        """
        if not 0 <= midi_note <= 127:
            raise ValueError(f"MIDI note must be in [0, 127], got {midi_note}")
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    @staticmethod
    def freq_to_note(freq: float) -> int:
        """Convert a frequency in Hz to the nearest MIDI note number.

        Parameters
        ----------
        freq : float
            Frequency in Hz. Must be positive.

        Returns
        -------
        int
            MIDI note number (0-127).

        Raises
        ------
        ValueError
            If freq is not positive.
        """
        if freq <= 0:
            raise ValueError(f"Frequency must be positive, got {freq}")
        midi = 69 + 12 * np.log2(freq / 440.0)
        return int(round(midi))

    @staticmethod
    def note_to_name(midi_note: int) -> str:
        """Convert a MIDI note number to its note name (e.g., 60 -> 'C4').

        Parameters
        ----------
        midi_note : int
            MIDI note number (0-127).

        Returns
        -------
        str
            Note name with octave number (e.g., 'C4').

        Raises
        ------
        ValueError
            If midi_note is outside the valid range [0, 127].
        """
        if not 0 <= midi_note <= 127:
            raise ValueError(f"MIDI note must be in [0, 127], got {midi_note}")
        name = MidiNotesUtil.NOTE_NAMES[midi_note % 12]
        octave = (midi_note // 12) - 1  # C4 = 60, so octave 4 for 60
        return f"{name}{octave}"

    @staticmethod
    def freq_to_bin(freq: float, sample_rate: int, fft_size: int) -> int:
        """Convert a frequency in Hz to the nearest FFT bin index.

        Parameters
        ----------
        freq : float
            Frequency in Hz.
        sample_rate : int
            Sample rate of the audio.
        fft_size : int
            FFT size.

        Returns
        -------
        int
            Bin index, clamped to the valid range.

        Raises
        ------
        ValueError
            If sample_rate or fft_size are non-positive.
        """
        if sample_rate <= 0 or fft_size <= 0:
            raise ValueError("sample_rate and fft_size must be positive")
        bin_index = int(round(freq * fft_size / sample_rate))
        max_bin = fft_size // 2
        return max(0, min(bin_index, max_bin))

    @staticmethod
    def midi_to_freq_bin(midi_note: int, sample_rate: int, fft_size: int) -> int:
        """Convert a MIDI note to the nearest FFT bin index.

        Convenience wrapper combining note_to_freq and freq_to_bin.

        Parameters
        ----------
        midi_note : int
            MIDI note number.
        sample_rate : int
            Sample rate of the audio.
        fft_size : int
            FFT size.

        Returns
        -------
        int
            Bin index.
        """
        freq = MidiNotesUtil.note_to_freq(midi_note)
        return MidiNotesUtil.freq_to_bin(freq, sample_rate, fft_size)
