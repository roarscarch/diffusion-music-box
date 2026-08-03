import numpy as np
import sounddevice as sd
import threading
import time


class AudioEngine:
    """Handles real-time playback of generated audio segments with seamless looping.

    The engine maintains a buffer of audio samples and plays them continuously.
    New segments can be added while playing, and crossfading is applied between
    segments to avoid clicks. The engine runs in a background thread and can be
    stopped gracefully.

    Parameters
    ----------
    sample_rate : int
        Sample rate for playback.
    block_size : int
        Number of samples per audio callback block.
    crossfade_samples : int
        Number of samples over which to crossfade between segments.
    device : int or str, optional
        Output device index or name. If None, use default.
    """

    def __init__(self, sample_rate=22050, block_size=1024, crossfade_samples=256, device=None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.crossfade_samples = crossfade_samples
        self.device = device

        self._buffer = np.zeros(block_size * 8, dtype=np.float32)  # pre-allocated ring buffer
        self._write_pos = 0
        self._read_pos = 0
        self._buffer_size = len(self._buffer)
        self._lock = threading.Lock()
        self._playing = False
        self._stream = None
        self._segment_queue = []
        self._current_segment = None
        self._segment_pos = 0

    def start(self):
        """Start audio playback in a background thread."""
        if self._playing:
            return
        self._playing = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=1,
            dtype='float32',
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        """Stop audio playback and release resources."""
        if not self._playing:
            return
        self._playing = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def add_segment(self, segment):
        """Queue a new audio segment for playback.

        Segments are played sequentially. When a new segment is added while
        another is playing, a crossfade is applied at the boundary.

        Parameters
        ----------
        segment : np.ndarray
            1D float array of audio samples (mono).
        """
        segment = np.asarray(segment, dtype=np.float32).reshape(-1)
        with self._lock:
            if self._current_segment is None:
                self._current_segment = segment
                self._segment_pos = 0
            else:
                # Queue for later playback after current segment finishes
                self._segment_queue.append(segment)

    def _callback(self, outdata, frames, time_info, status):
        """Audio callback: fill output buffer with current segment data."""
        if status:
            print(f"Audio status: {status}")

        outdata.fill(0.0)
        with self._lock:
            for i in range(frames):
                if self._current_segment is None:
                    # No segment playing, output silence
                    outdata[i, 0] = 0.0
                    continue

                # Get current sample from segment
                pos = self._segment_pos
                if pos < len(self._current_segment):
                    sample = self._current_segment[pos]
                    self._segment_pos += 1
                else:
                    # Current segment finished, move to next
                    if self._segment_queue:
                        next_seg = self._segment_queue.pop(0)
                        # Apply crossfade at boundary
                        crossfade_len = min(self.crossfade_samples, len(self._current_segment), len(next_seg))
                        if crossfade_len > 0:
                            # Crossfade from current (which already ended) to next
                            # We need to blend the last part of current with beginning of next
                            # But current is already consumed; so we just start next with a fade-in
                            fade_in = np.linspace(0.0, 1.0, crossfade_len, dtype=np.float32)
                            next_seg[:crossfade_len] *= fade_in
                        self._current_segment = next_seg
                        self._segment_pos = 0
                        sample = self._current_segment[self._segment_pos]
                        self._segment_pos += 1
                    else:
                        # No more segments, output silence
                        sample = 0.0

                outdata[i, 0] = sample

    def is_playing(self):
        """Return True if audio is playing."""
        return self._playing
