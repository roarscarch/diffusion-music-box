import numpy as np
import threading
import time
from src.diffusion_model import DiffusionModel
from src.spectrogram import Spectrogram
from src.spectrum_inverse import SpectrumInverse
from src.crossfade_mixer import CrossfadeMixer
from src.audio_buffer import AudioSegmentBuffer
from src.audio_engine import AudioEngine


class RealTimeGenerator:
    """Generate audio in real time from a diffusion model on spectrograms.

    This class orchestrates the entire pipeline: it maintains a diffusion model
    that operates on a fixed-size frequency-time tile, generates new tiles
    continuously, converts them back to audio via an inverse spectrogram
    transform, and feeds the resulting segments into an audio engine for
    seamless playback with crossfading.

    The generator runs in a background thread, producing segments as fast as
    they are consumed by the audio engine. Parameters such as the number of
    diffusion steps and the noise schedule can be adjusted on the fly.

    Parameters
    ----------
    sample_rate : int
        Sample rate for the generated audio.
    fft_size : int
        FFT size for the spectrogram representation.
    hop_length : int
        Hop length in samples between time frames.
    tile_width : int
        Number of time frames in each generated spectrogram tile.
    diffusion_steps : int
        Number of iterative denoising steps per tile.
    crossfade_samples : int
        Number of samples over which to crossfade between segments.
    device : int or str, optional
        Output device for the audio engine.
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_length=256,
                 tile_width=64, diffusion_steps=20, crossfade_samples=256,
                 device=None):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.tile_width = tile_width
        self.diffusion_steps = diffusion_steps
        self.crossfade_samples = crossfade_samples
        self.device = device

        self.freq_bins = fft_size // 2 + 1
        self.segment_length = tile_width * hop_length

        # Core components
        self.spectrogram = Spectrogram(sample_rate, fft_size, hop_length)
        self.inverse = SpectrumInverse(sample_rate, fft_size, hop_length)
        self.diffusion = DiffusionModel(
            input_shape=(self.freq_bins, self.tile_width),
            steps=diffusion_steps
        )
        self.mixer = CrossfadeMixer(crossfade_samples)
        self.buffer = AudioSegmentBuffer(max_segments=16)
        self.engine = AudioEngine(
            sample_rate=sample_rate,
            block_size=hop_length * 2,
            crossfade_samples=crossfade_samples,
            device=device
        )

        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._noise_scale = 1.0
        self._tempo = 60.0
        self._last_tile = None  # previous tile for smooth transitions

    def set_diffusion_steps(self, steps):
        """Set the number of diffusion steps for subsequent tiles."""
        with self._lock:
            self.diffusion_steps = max(1, int(steps))

    def set_noise_scale(self, scale):
        """Set the noise scale for the initial random tile."""
        with self._lock:
            self._noise_scale = max(0.0, float(scale))

    def set_tempo(self, bpm):
        """Set the tempo (affects the overlap-add time stretching)."""
        with self._lock:
            self._tempo = max(20.0, min(200.0, float(bpm)))

    def start(self):
        """Start the generation and playback loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.engine.start()

    def stop(self):
        """Stop the generation and playback loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.engine.stop()

    def _run(self):
        """Main generation loop: produce segments and feed to the buffer."""
        rng = np.random.default_rng()
        while self._running:
            # Check if the engine needs more data
            if not self.engine.needs_more():
                time.sleep(0.01)
                continue

            # Generate a new spectrogram tile
            tile = self._generate_tile(rng)

            # Convert tile to audio segment
            audio = self.inverse.spectrogram_to_audio(tile)

            # Apply crossfade with the previous segment (handled in engine?)
            # The engine already crossfades, so just add to buffer
            self.buffer.add(audio)

            # Feed the engine directly (the engine pulls from buffer)
            self.engine.add_segment(audio)

            # Store for next iteration (could be used for conditioning)
            self._last_tile = tile

            # Small sleep to avoid busy-waiting
            time.sleep(0.005)

    def _generate_tile(self, rng):
        """Generate a single spectrogram tile using the diffusion model."""
        # Create a random noise tile as the starting point
        noise = rng.standard_normal((self.freq_bins, self.tile_width)).astype(np.float32)
        noise *= self._noise_scale

        # Optionally, use the previous tile as a prior (for smoother evolution)
        if self._last_tile is not None:
            # Blend a small amount of the previous tile into the noise
            # This creates a temporal continuity between tiles
            blend = 0.3
            noise = (1 - blend) * noise + blend * self._last_tile

        # Run the diffusion process (denoising)
        with self._lock:
            steps = self.diffusion_steps
        tile = self.diffusion.denoise(noise, steps=steps)

        # Normalize to a reasonable range
        tile = np.clip(tile, -1.0, 1.0)
        return tile

    def _generate_continuation(self, previous_tile, rng):
        """Generate a tile that continues from the previous one."""
        # Simple approach: start from a noise that is conditioned on previous tile
        noise = rng.standard_normal((self.freq_bins, self.tile_width)).astype(np.float32)
        # Add a small fraction of the previous tile's last column as a seed
        if previous_tile is not None:
            seed = previous_tile[:, -1:]
            # Repeat the seed across the tile to create a smooth continuation
            seed_full = np.repeat(seed, self.tile_width, axis=1)
            noise = 0.5 * noise + 0.5 * seed_full
        return self.diffusion.denoise(noise, steps=self.diffusion_steps)
