import numpy as np
import time
import threading
from src.scheduler import SpectrogramScheduler


class TempoScheduler:
    """Schedule generation of spectrogram tiles based on a tempo.

    This scheduler extends the base SpectrogramScheduler to generate
    new tiles at a rate determined by the tempo (beats per minute).
    Each tile corresponds to a fixed number of beats, and generation
    is triggered when the playback position reaches the end of the
    current tile's time span. This creates a musical pulse that aligns
    generated segments with a tempo.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio.
    n_freq : int
        Number of frequency bins in the spectrogram.
    n_frames : int
        Number of time frames per tile.
    hop_length : int, optional
        Hop length used in the spectrogram transformation.
    tempo : float, optional
        Initial tempo in beats per minute.
    beats_per_tile : int, optional
        Number of beats each tile spans.
    """

    def __init__(self, sample_rate=22050, n_freq=257, n_frames=128, hop_length=256, tempo=120.0, beats_per_tile=4):
        super().__init__(sample_rate, n_freq, n_frames, hop_length)
        self.tempo = tempo
        self.beats_per_tile = beats_per_tile
        self._lock = threading.Lock()
        self._next_tile_audio_time = 0.0

    @property
    def seconds_per_beat(self):
        return 60.0 / self.tempo

    @property
    def tile_seconds(self):
        return self.seconds_per_beat * self.beats_per_tile

    def set_tempo(self, tempo):
        """Update the tempo in real time.

        The scheduler will pick up the new tempo on the next generation
        trigger, so changes take effect within one tile duration.

        Parameters
        ----------
        tempo : float
            New tempo in beats per minute. Must be > 0.
        """
        if tempo <= 0:
            raise ValueError("Tempo must be positive")
        with self._lock:
            self.tempo = float(tempo)

    def is_tile_due(self, audio_position):
        """Check whether a new tile should be generated now.

        Parameters
        ----------
        audio_position : float
            Current playback position in seconds.

        Returns
        -------
        bool
            True if a new tile generation should be triggered.
        """
        with self._lock:
            tile_seconds = self.tile_seconds
            if audio_position < self._next_tile_audio_time:
                return False
            # Advance to the next boundary
            self._next_tile_audio_time += tile_seconds
            return True

    def reset_timing(self):
        """Reset the internal timing to start at the current position."""
        with self._lock:
            self._next_tile_audio_time = 0.0

    def generate_tile(self, generator):
        """Generate a new tile using the provided generator.

        Parameters
        ----------
        generator : callable
            A callable that returns a spectrogram tile as a numpy array.

        Returns
        -------
        numpy.ndarray
            The generated tile.
        """
        return generator()
