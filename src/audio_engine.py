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
        self._segment_offset = 0
        self._crossfade_buffer = None
        self._crossfade_pos = 0
        self._crossfade_active = False

    def start(self):
        """Start the audio stream in a background thread."""
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
        """Stop the audio stream gracefully."""
        if not self._playing:
            return
        self._playing = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def add_segment(self, segment):
        """Add a new audio segment to be played after the current one.

        If a segment is already queued, the new segment is appended. The engine
        will crossfade from the current segment to the next one when the queue
        is consumed.

        Parameters
        ----------
        segment : np.ndarray
            1D float array of audio samples.
        """
        segment = np.asarray(segment, dtype=np.float32)
        if segment.ndim != 1:
            raise ValueError("Segment must be 1D")
        with self._lock:
            self._segment_queue.append(segment)

    def _callback(self, outdata, frames, time_info, status):
        """Audio callback: fill output buffer with samples."""
        if status:
            print(f"Audio callback status: {status}")

        outdata[:, 0] = 0.0

        with self._lock:
            # Start crossfade if we have a new segment and not already crossfading
            if not self._crossfade_active and self._segment_queue:
                self._start_crossfade_locked()

            # Write samples from the current segment or crossfade buffer
            for i in range(frames):
                if self._crossfade_active:
                    # Crossfade mode: blend current segment and next segment
                    if self._crossfade_pos < self.crossfade_samples:
                        # Blending
                        alpha = self._crossfade_pos / self.crossfade_samples
                        current_sample = self._get_current_sample_locked()
                        next_sample = self._get_next_sample_locked()
                        outdata[i, 0] = (1.0 - alpha) * current_sample + alpha * next_sample
                        self._crossfade_pos += 1
                    else:
                        # Crossfade done, switch to next segment
                        self._finish_crossfade_locked()
                        outdata[i, 0] = self._get_current_sample_locked()
                else:
                    # Normal playback
                    outdata[i, 0] = self._get_current_sample_locked()

    def _start_crossfade_locked(self):
        """Start crossfading from current segment to the next queued segment."""
        if not self._segment_queue:
            return
        next_segment = self._segment_queue.pop(0)
        self._crossfade_buffer = next_segment
        self._crossfade_pos = 0
        self._crossfade_active = True

    def _finish_crossfade_locked(self):
        """Finish crossfade: set current segment to the crossfade buffer."""
        self._current_segment = self._crossfade_buffer
        self._segment_offset = 0
        self._crossfade_buffer = None
        self._crossfade_pos = 0
        self._crossfade_active = False

    def _get_current_sample_locked(self):
        """Get the next sample from the current segment, handling looping."""
        if self._current_segment is None or len(self._current_segment) == 0:
            return 0.0
        sample = self._current_segment[self._segment_offset]
        self._segment_offset += 1
        if self._segment_offset >= len(self._current_segment):
            # Loop back to start of the same segment if no queue
            if not self._segment_queue:
                self._segment_offset = 0
            else:
                # If there is a queue, we'll switch on next callback
                self._segment_offset = 0
        return float(sample)

    def _get_next_sample_locked(self):
        """Get the next sample from the crossfade buffer."""
        if self._crossfade_buffer is None or len(self._crossfade_buffer) == 0:
            return 0.0
        pos = self._crossfade_pos
        if pos >= len(self._crossfade_buffer):
            return 0.0
        return float(self._crossfade_buffer[pos])

    def clear_queue(self):
        """Clear any queued segments."""
        with self._lock:
            self._segment_queue.clear()

    @property
    def is_playing(self):
        """Return True if the engine is currently playing."""
        return self._playing
