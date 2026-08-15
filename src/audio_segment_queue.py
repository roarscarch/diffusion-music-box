import numpy as np
import threading


class AudioSegmentQueue:
    """A thread-safe queue for audio segments with crossfade support.

    This class manages a queue of audio segments that can be added from the
    generation thread and consumed by the audio engine. It supports
    concatenating segments with optional crossfading to ensure seamless
    playback between segments. The queue is bounded to prevent unbounded
    memory growth.

    Parameters
    ----------
    max_segments : int
        Maximum number of segments to store in the queue.
    crossfade_samples : int, optional
        Number of samples over which to crossfade between adjacent segments.
        If zero, segments are concatenated without crossfading.
    """

    def __init__(self, max_segments=16, crossfade_samples=0):
        self._segments = []
        self._lock = threading.Lock()
        self._max_segments = max_segments
        self._crossfade_samples = crossfade_samples

    def add(self, segment):
        """Add an audio segment to the queue.

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
        """Remove and return the oldest segment from the queue.

        Returns
        -------
        np.ndarray or None
            The oldest segment as a 1D float array, or None if the queue is empty.
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
            The oldest segment as a 1D float array, or None if the queue is empty.
        """
        with self._lock:
            if not self._segments:
                return None
            return self._segments[0]

    def clear(self):
        """Remove all segments from the queue."""
        with self._lock:
            self._segments.clear()

    def __len__(self):
        """Return the number of segments currently in the queue."""
        with self._lock:
            return len(self._segments)

    def is_empty(self):
        """Return True if the queue is empty."""
        with self._lock:
            return len(self._segments) == 0

    def get_all(self):
        """Return a copy of all segments in the queue.

        Returns
        -------
        list of np.ndarray
            A list of all segments in the queue.
        """
        with self._lock:
            return list(self._segments)

    def concatenate_with_crossfade(self):
        """Concatenate all segments in the queue into a single array.

        If crossfade_samples is greater than zero, adjacent segments are
        blended over the crossfade region. If the queue is empty, returns an
        empty array. The queue is cleared after concatenation.

        Returns
        -------
        np.ndarray
            The concatenated audio samples as a 1D float array.
        """
        with self._lock:
            if not self._segments:
                return np.zeros(0, dtype=np.float32)
            if len(self._segments) == 1:
                result = self._segments[0].copy()
            else:
                # Concatenate with crossfade
                fade_len = min(self._crossfade_samples, len(self._segments[0]), len(self._segments[-1]))
                if fade_len <= 0:
                    # No crossfade needed
                    result = np.concatenate(self._segments)
                else:
                    # Build output array with room for crossfades
                    total_len = sum(len(seg) for seg in self._segments) - fade_len * (len(self._segments) - 1)
                    result = np.zeros(total_len, dtype=np.float32)
                    pos = 0
                    for i, seg in enumerate(self._segments):
                        if i == 0:
                            # First segment: copy whole
                            result[:len(seg)] = seg
                            pos = len(seg)
                        else:
                            # Crossfade with previous segment
                            # Previous segment's tail is already in result at pos - len(prev_seg)
                            prev_len = len(self._segments[i-1])
                            # Overlap region length
                            overlap = min(fade_len, prev_len, len(seg))
                            # Apply crossfade to overlap
                            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                            fade_out = 1.0 - fade_in
                            # Add the fade-in of current segment to the tail of previous
                            result[pos - overlap:pos] *= fade_out
                            result[pos - overlap:pos] += seg[:overlap] * fade_in
                            # Copy the rest of the current segment
                            result[pos:pos + len(seg) - overlap] = seg[overlap:]
                            pos += len(seg) - overlap
            self._segments.clear()
            return result
