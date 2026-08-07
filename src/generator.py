import numpy as np
from src.spectrogram import SpectrogramGenerator
from src.denoiser import Denoiser
from src.noise_schedule import NoiseSchedule
from src.spectrum_inverse import SpectrumInverse
from src.overlap_add import OverlapAdd
from src.audio_buffer import AudioSegmentBuffer


class TileGenerator:
    """Generates spectrogram tiles via iterative diffusion and converts them to audio.

    This class encapsulates the core generation loop: it takes a noise schedule,
    a denoiser, and an inverse transform, and produces audio segments from
    random spectrogram tiles. It supports variable number of diffusion steps
    and a temperature parameter to control randomness.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        noise_schedule: NoiseSchedule,
        inverse: SpectrumInverse,
        sample_rate: int = 22050,
        n_fft: int = 512,
        hop_length: int = 128,
        tile_width: int = 64,
        tile_height: int = 256,
        seed: int = None,
    ):
        self.denoiser = denoiser
        self.noise_schedule = noise_schedule
        self.inverse = inverse
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.rng = np.random.default_rng(seed)

    def generate_tile(self, steps: int = 50, temperature: float = 1.0) -> np.ndarray:
        """Generate a single spectrogram tile using iterative diffusion.

        Parameters
        ----------
        steps : int
            Number of diffusion steps.
        temperature : float
            Scaling factor applied to the noise at each step. Higher values
            increase randomness.

        Returns
        -------
        np.ndarray
            Generated spectrogram tile with shape (tile_height, tile_width).
        """
        # Start from pure noise
        tile = self.rng.standard_normal((self.tile_height, self.tile_width)).astype(np.float32)

        # Iteratively denoise
        for t in range(steps):
            alpha = self.noise_schedule.get_alpha(t, steps)
            # Predict the clean tile using the denoiser
            predicted = self.denoiser.denoise(tile, alpha)
            # Add scaled noise for stochasticity
            noise = self.rng.standard_normal(tile.shape).astype(np.float32)
            tile = predicted + temperature * noise * (1.0 - alpha)

        # Normalize to [0, 1] range for spectrogram
        tile = (tile - tile.min()) / (tile.max() - tile.min() + 1e-8)
        return tile

    def generate_segment(self, steps: int = 50, temperature: float = 1.0) -> np.ndarray:
        """Generate a full audio segment from a spectrogram tile.

        Parameters
        ----------
        steps : int
            Number of diffusion steps.
        temperature : float
            Scaling factor for noise during diffusion.

        Returns
        -------
        np.ndarray
            Audio samples as a 1D float32 array.
        """
        tile = self.generate_tile(steps=steps, temperature=temperature)
        # Convert spectrogram tile to audio using inverse transform
        audio = self.inverse.inverse_transform(tile)
        # Normalize to avoid clipping
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio.astype(np.float32)

    def generate_stream(
        self,
        buffer: AudioSegmentBuffer,
        steps: int = 50,
        temperature: float = 1.0,
        num_segments: int = 8,
    ):
        """Generate multiple segments and add them to a buffer.

        This is a convenience method for continuous generation.

        Parameters
        ----------
        buffer : AudioSegmentBuffer
            Buffer to add generated segments to.
        steps : int
            Number of diffusion steps per tile.
        temperature : float
            Noise scaling factor.
        num_segments : int
            Number of segments to generate.
        """
        for _ in range(num_segments):
            segment = self.generate_segment(steps=steps, temperature=temperature)
            buffer.add(segment)
