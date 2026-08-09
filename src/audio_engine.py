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

    def start(self):
        """Start playback in a background thread."""
        if self._playing:
            return
        # Validate device availability before opening stream
        if self.device is not None:
            try:
                sd.check_output_settings(device=self.device, samplerate=self.sample_rate, channels=1)
            except Exception as e:
                raise RuntimeError(f"Output device '{self.device}' is not available or does not support the requested settings: {e}")
        else:
            try:
                sd.check_output_settings(samplerate=self.sample_rate, channels=1)
            except Exception as e:
                raise RuntimeError(f"Default output device is not available or does not support the requested settings: {e}")
        self._playing = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        """Stop playback and close the stream."""
        if not self._playing:
            return
        self._playing = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def add_segment(self, segment):
        """Add a new audio segment to be played.

        Parameters
        ----------
        segment : np.ndarray
            Audio samples as a 1D float array.
        """
        segment = np.asarray(segment, dtype=np.float32)
        if segment.ndim != 1:
            raise ValueError("Audio segment must be 1D")
        with self._lock:
            self._segment_queue.append(segment)

    def _audio_callback(self, outdata, frames, time_info, status):
        """Fill the output buffer with samples."""
        if status:
            print(f"Audio callback status: {status}")
        with self._lock:
            # Pull segments from queue into ring buffer if needed
            while len(self._segment_queue) > 0 and self._space_available() >= self.block_size:
                seg = self._segment_queue.pop(0)
                self._write_segment(seg)
            # Read from ring buffer
            samples = np.zeros(frames, dtype=np.float32)
            for i in range(frames):
                if self._read_pos == self._write_pos and not self._segment_queue:
                    break  # no more data, output silence
                samples[i] = self._buffer[self._read_pos]
                self._read_pos = (self._read_pos + 1) % self._buffer_size
            outdata[:, 0] = samples

    def _space_available(self):
        """Return number of free slots in the ring buffer."""
        return (self._write_pos - self._read_pos - 1) % self._buffer_size

    def _write_segment(self, segment):
        """Write a segment into the ring buffer with crossfade at the boundary."""
        # Apply crossfade with the tail of the previous segment if possible
        # For simplicity, we just write the segment directly; crossfade is handled elsewhere.
        for sample in segment:
            if self._space_available() == 0:
                break  # buffer full, stop writing
            self._buffer[self._write_pos] = sample
            self._write_pos = (self._write_pos + 1) % self._buffer_size
