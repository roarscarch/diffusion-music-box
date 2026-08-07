import numpy as np


class TempoScheduler:
    """Schedules audio segments according to a tempo (BPM) with beat alignment.

    This module provides a simple tempo scheduler that determines when to trigger
    new audio segments based on the current tempo. It supports dynamic tempo
    changes and can be used to synchronize generated segments with a musical
    beat, ensuring a consistent rhythmic feel.

    Parameters
    ----------
    bpm : float
        Beats per minute. Must be positive.
    beats_per_segment : int
        Number of beats per audio segment. Determines segment duration.
    sample_rate : int
        Sample rate used for time calculations.
    """

    def __init__(self, bpm=60.0, beats_per_segment=4, sample_rate=22050):
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        if beats_per_segment <= 0:
            raise ValueError("beats_per_segment must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.bpm = bpm
        self.beats_per_segment = beats_per_segment
        self.sample_rate = sample_rate
        self._beat_duration = 60.0 / bpm  # seconds per beat
        self._segment_duration = self._beat_duration * beats_per_segment

    def update_bpm(self, bpm):
        """Update the tempo and recalculate durations.

        Parameters
        ----------
        bpm : float
            New beats per minute. Must be positive.
        """
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        self.bpm = bpm
        self._beat_duration = 60.0 / bpm
        self._segment_duration = self._beat_duration * self.beats_per_segment

    def segment_length_samples(self):
        """Return the length of a segment in samples.

        Returns
        -------
        int
            Number of samples in one segment.
        """
        return int(round(self._segment_duration * self.sample_rate))

    def beat_times(self, num_beats):
        """Return the sample positions of beat boundaries.

        Parameters
        ----------
        num_beats : int
            Number of beats to generate positions for.

        Returns
        -------
        np.ndarray
            Sample indices at which beats occur.
        """
        if num_beats <= 0:
            raise ValueError("num_beats must be positive")
        # Beat boundaries are at multiples of beat duration.
        # First beat at sample 0, then every beat_duration samples.
        return np.arange(num_beats, dtype=np.int64) * int(round(self._beat_duration * self.sample_rate))

    def next_beat_sample(self, current_sample):
        """Return the sample position of the next beat boundary.

        Parameters
        ----------
        current_sample : int
            Current sample position.

        Returns
        -------
        int
            Sample position of the next beat boundary after current_sample.
        """
        if current_sample < 0:
            raise ValueError("current_sample must be non-negative")
        beat_duration_samples = int(round(self._beat_duration * self.sample_rate))
        if beat_duration_samples == 0:
            return current_sample
        # Ceiling division to get the next multiple.
        next_beat = ((current_sample // beat_duration_samples) + 1) * beat_duration_samples
        return next_beat

    def is_beat(self, sample):
        """Check if a given sample position is a beat boundary.

        Parameters
        ----------
        sample : int
            Sample position to check.

        Returns
        -------
        bool
            True if the sample is exactly on a beat boundary.
        """
        beat_duration_samples = int(round(self._beat_duration * self.sample_rate))
        if beat_duration_samples == 0:
            return False
        return sample % beat_duration_samples == 0
}