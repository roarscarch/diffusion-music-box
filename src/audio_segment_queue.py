import threading
import numpy as np


class AudioSegmentQueue:
    """A thread-safe queue for audio segments with optional overlap and crossfade.

    This class provides a FIFO queue for audio segments. It is designed to be
    used by a producer (e.g., a real-time generator) and a consumer (e.g.,
    an audio engine). The queue supports adding segments and retrieving them
    in order, with optional overlap and crossfade blending between consecutive
    segments to ensure smooth transitions.

    The queue is bounded to prevent unbounded memory growth. When full, the
    oldest segment is dropped to make room for new ones.

    Parameters
    ----------
    max_segments : int
        Maximum number of segments to store in the queue.
    overlap_samples : int
        Number of samples that consecutive segments should overlap. When
        retrieving a segment, the overlap region is blended with the next
        segment using a linear crossfade.
    """

    def __init__(self, max_segments=16, overlap_samples=0):
        self._segments = []
        self._lock = threading.Condition()
        self._max_segments = max_segments
        self._overlap = int(overlap_samples)
        if self._overlap < 0:
            raise ValueError("overlap_samples must be non-negative")

    def put(self, segment):
        """Add a segment to the queue.

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
            self._lock.notify()

    def get(self, block=True, timeout=None):
        """Get the next segment from the queue, optionally with crossfade.

        When there is at least one segment available, the first segment is
        returned. If there is a next segment and overlap_samples > 0, the
        overlapping tail of the first segment is crossfaded with the head of
        the next segment, and the next segment is trimmed accordingly.

        Parameters
        ----------
        block : bool
            If True, block until a segment is available. If False, return
            None immediately if the queue is empty.
        timeout : float or None
            Maximum time to block in seconds. Only used if block is True.

        Returns
        -------
        np.ndarray or None
            The next audio segment as a 1D float array, or None if no segment
            is available (when block=False or timeout expires).
        """
        with self._lock:
            if not block:
                if not self._segments:
                    return None
            else:
                if not self._segments:
                    self._lock.wait(timeout)
                if not self._segments:
                    return None

            seg = self._segments.pop(0)

            # If overlap is enabled and there is a next segment, blend them
            if self._overlap > 0 and self._segments:
                next_seg = self._segments[0]
                if len(seg) >= self._overlap and len(next_seg) >= self._overlap:
                    # Crossfade the tail of seg with the head of next_seg
                    fade_in = np.linspace(0.0, 1.0, self._overlap, dtype=np.float32)
                    fade_out = 1.0 - fade_in
                    tail = seg[-self._overlap:] * fade_out
                    head = next_seg[:self._overlap] * fade_in
                    blended = tail + head
                    # Replace the overlap region in the next segment with the
                    # blended version (this will be used later)
                    next_seg = np.concatenate([
                        blended,
                        next_seg[self._overlap:]
                    ])
                    self._segments[0] = next_seg
                    # Trim the returned segment to exclude the overlap region
                    seg = seg[:-self._overlap]

            return seg

    def qsize(self):
        """Return the current number of segments in the queue."""
        with self._lock:
            return len(self._segments)

    def empty(self):
        """Return True if the queue is empty."""
        with self._lock:
            return len(self._segments) == 0

    def clear(self):
        """Remove all segments from the queue."""
        with self._lock:
            self._segments.clear()
            self._lock.notify_all()
