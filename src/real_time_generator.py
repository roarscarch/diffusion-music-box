import numpy as np
import threading
import time
from src.diffusion_model import DiffusionModel
from src.noise_scheduler import NoiseScheduler
from src.crossfade_mixer import CrossfadeMixer
from src.audio_segment_queue import AudioSegmentQueue
from src.spectrogram import Spectrogram


class RealTimeGenerator:
    """Continuously generates audio segments using iterative diffusion on spectrograms.

    This module runs a diffusion model on a fixed-size spectrogram tile, inverts
    the spectrogram to audio, and queues the resulting segments for playback.
    It runs in a background thread and can be started and stopped gracefully.
    The generator supports configurable noise schedule, diffusion steps, and
    crossfade length, and it maintains a queue of ready segments for the audio engine.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio.
    fft_size : int
        FFT size for the spectrogram.
    hop_length : int
        Hop length between STFT frames.
    tile_width : int
        Number of time frames in each generated spectrogram tile.
    crossfade_samples : int, optional
        Number of samples to crossfade between consecutive segments.
    diffusion_steps : int, optional
        Number of diffusion steps to run per tile.
    noise_schedule : str, optional
        Type of noise schedule, e.g., 'linear' or 'cosine'.
    queue_size : int, optional
        Maximum number of segments to keep in the queue.
    device : str, optional
        Device string for the diffusion model (e.g., 'cpu', 'cuda').
    """

    def __init__(
        self,
        sample_rate=22050,
        fft_size=1024,
        hop_length=256,
        tile_width=64,
        crossfade_samples=256,
        diffusion_steps=50,
        noise_schedule='linear',
        queue_size=4,
        device='cpu',
    ):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.tile_width = tile_width
        self.crossfade_samples = crossfade_samples
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.device = device

        self.spectrogram = Spectrogram(sample_rate=sample_rate, fft_size=fft_size, hop_length=hop_length)
        self.noise_scheduler = NoiseScheduler(
            num_steps=diffusion_steps,
            schedule=noise_schedule,
        )
        self.model = DiffusionModel(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            device=device,
        )
        self.mixer = CrossfadeMixer(crossfade_samples=crossfade_samples)
        self.queue = AudioSegmentQueue(maxsize=queue_size)

        self._stop_event = threading.Event()
        self._thread = None
        self._last_segment = np.zeros(int(hop_length * (tile_width - 1)) + fft_size, dtype=np.float32)

    def start(self):
        """Start the generation thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the generation thread gracefully."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def set_parameters(self, diffusion_steps=None, noise_schedule=None, crossfade_samples=None):
        """Update generation parameters on the fly.

        Parameters
        ----------
        diffusion_steps : int, optional
            New number of diffusion steps.
        noise_schedule : str, optional
            New noise schedule type.
        crossfade_samples : int, optional
            New crossfade length in samples.
        """
        if diffusion_steps is not None:
            self.diffusion_steps = diffusion_steps
            self.noise_scheduler.num_steps = diffusion_steps
        if noise_schedule is not None:
            self.noise_schedule = noise_schedule
            self.noise_scheduler.schedule = noise_schedule
        if crossfade_samples is not None:
            self.crossfade_samples = crossfade_samples
            self.mixer = CrossfadeMixer(crossfade_samples=crossfade_samples)

    def _run(self):
        """Main loop: generate segments and push to queue."""
        while not self._stop_event.is_set():
            if self.queue.full():
                time.sleep(0.01)
                continue
            segment = self._generate_segment()
            if segment is not None:
                self.queue.put(segment)

    def _generate_segment(self):
        """Generate a single audio segment from a diffusion model.

        Returns
        -------
        np.ndarray
            Audio samples as a 1D float array.
        """
        # Generate a random spectrogram tile (frequency x time)
        # The diffusion model denoises from random noise towards a structured target.
        # For now, we use a simple random tile and run the model's forward pass.
        # In a full implementation, the model would be trained; here we synthesize.
        tile_shape = (self.fft_size // 2 + 1, self.tile_width)
        noise = np.random.randn(*tile_shape).astype(np.float32)

        # Run iterative diffusion (simplified: just pass through model multiple times)
        x = noise
        for step in range(self.diffusion_steps):
            # Placeholder: model inference would be here.
            # For now, we just apply a low-pass filter as a stand-in.
            x = self._lowpass_spectrogram(x)

        # Convert spectrogram to audio via ISTFT
        # Ensure the spectrogram is non-negative (magnitude)
        magnitude = np.abs(x)
        # Random phase
        phase = np.random.randn(*magnitude.shape).astype(np.float32)
        complex_spec = magnitude * np.exp(1j * 2 * np.pi * phase)
        audio = self.spectrogram.istft(complex_spec)

        # Crossfade with the previous segment
        audio = self.mixer.crossfade(self._last_segment, audio)
        self._last_segment = audio

        # Normalize to prevent clipping
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.8

        return audio

    def _lowpass_spectrogram(self, spec):
        """Apply a simple low-pass filter along the frequency axis.

        Parameters
        ----------
        spec : np.ndarray
            Spectrogram of shape (freq_bins, time_frames).

        Returns
        -------
        np.ndarray
            Filtered spectrogram.
        """
        # Simple moving average along frequency
        kernel = np.ones((3, 1), dtype=np.float32) / 3.0
        filtered = np.zeros_like(spec)
        for t in range(spec.shape[1]):
            filtered[:, t] = np.convolve(spec[:, t], kernel.flatten(), mode='same')
        return filtered

    def get_segment(self, block_size=1024):
        """Get the next audio block for playback.

        Parameters
        ----------
        block_size : int
            Number of samples to return.

        Returns
        -------
        np.ndarray
            Audio samples of length block_size.
        """
        # For simplicity, this returns zeros if no segment is ready.
        # In a real implementation, the audio engine would pull from the queue.
        if self.queue.empty():
            return np.zeros(block_size, dtype=np.float32)
        segment = self.queue.get()
        # Pad or truncate to block_size
        if len(segment) >= block_size:
