import numpy as np
import threading


class AudioSegmentBuffer:
    """A thread-safe buffer for storing and retrieving audio segments.

    This class provides a simple queue-like structure for audio segments,
    allowing the audio engine to pull segments in order while new segments
    are added from the generation thread. It uses a lock to ensure thread
    safety and supports clearing the buffer.
    """

    def __init__(self, max_segments=16):
        self._segments = []
        self._lock = threading.Lock()
        self._max_segments = max_segments

    def add(self, segment):
        """Add an audio segment to the buffer.

        Parameters
        ----------
        segment : np.ndarray
            Audio samples as a 1D float array.

        Raises
        ------
        ValueError
            If the segment is not a 1D float array.
        """
        segment = np.asarray(segment, dtype=np.float32)
        if segment.ndim != 1:
            raise ValueError("Audio segment must be 1D")
        with self._lock:
            self._segments.append(segment)
            if len(self._segments) > self._max_segments:
                # Drop the oldest segment to prevent unbounded growth
                self._segments.pop(0)

    def pop(self):
        """Remove and return the oldest segment from the buffer.

        Returns
        -------
        np.ndarray or None
            The oldest segment, or None if the buffer is empty.
        """
        with self._lock:
            if not self._segments:
                return None
            return self._segments.pop(0)

    def peek(self):
        """Return the oldest segment without removing it.

        Returns
        -------
        np.ndarray or None
            The oldest segment, or None if the buffer is empty.
        """
        with self._lock:
            if not self._segments:
                return None
            return self._segments[0]

    def __len__(self):
        """Return the number of segments currently in the buffer."""
        with self._lock:
            return len(self._segments)

    def clear(self):
        """Remove all segments from the buffer."""
        with self._lock:
            self._segments.clear()

    def is_empty(self):
        """Return True if the buffer has no segments."""
        with self._lock:
            return len(self._segments) == 0
