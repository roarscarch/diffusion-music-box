import numpy as np
from .denoiser import Denoiser
from .diffusion_model import DiffusionModel
from .crossfade_mixer import CrossfadeMixer
from .overlap_add import OverlapAdd
from .noise_schedule import NoiseSchedule


class TilePipeline:
    """Orchestrates the generation of audio from spectrogram tiles.

    This pipeline ties together the diffusion model, denoiser, crossfade
    mixer, and overlap-add to produce seamless, evolving ambient audio.
    It generates a sequence of spectrogram tiles using the diffusion model,
    converts them to audio, and blends them with crossfade to avoid clicks.
    The pipeline is designed for real-time use, generating tiles on demand.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio.
    fft_size : int
        FFT size for the spectrogram.
    hop_length : int
        Hop length between frames.
    diffusion_steps : int
        Number of diffusion steps to run per tile.
    noise_schedule : NoiseSchedule or None
        Schedule for noise levels during diffusion.
    crossfade_samples : int
        Number of samples to crossfade between tiles.
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_length=256,
                 diffusion_steps=50, noise_schedule=None, crossfade_samples=256):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.diffusion_steps = diffusion_steps
        self.crossfade_samples = crossfade_samples

        self.noise_schedule = noise_schedule or NoiseSchedule('linear')
        self.denoiser = Denoiser(fft_size=fft_size)
        self.diffusion_model = DiffusionModel(
            denoiser=self.denoiser,
            noise_schedule=self.noise_schedule,
            sample_rate=sample_rate,
            fft_size=fft_size,
            hop_length=hop_length
        )
        self.crossfade_mixer = CrossfadeMixer(crossfade_samples=crossfade_samples)
        self.overlap_add = OverlapAdd(hop_length=hop_length)

    def generate_tile(self, seed=None):
        """Generate a single spectrogram tile using the diffusion model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            2D float array of shape (freq_bins, time_steps) representing the
            spectrogram tile.
        """
        return self.diffusion_model.generate(seed=seed)

    def tile_to_audio(self, tile):
        """Convert a spectrogram tile to audio using inverse STFT.

        Parameters
        ----------
        tile : np.ndarray
            2D spectrogram tile of shape (freq_bins, time_steps).

        Returns
        -------
        np.ndarray
            1D float array of audio samples.
        """
        # Use the diffusion model's inverse transform (spectrum_inverse)
        return self.diffusion_model.spectrum_inverse(tile)

    def process(self, num_tiles=1, seed=None):
        """Generate a sequence of audio segments and blend them.

        Parameters
        ----------
        num_tiles : int
            Number of tiles to generate.
        seed : int, optional
            Base seed for reproducibility; each tile uses seed + index.

        Returns
        -------
        np.ndarray
            Concatenated audio with crossfade applied between segments.
        """
        audio_segments = []
        for i in range(num_tiles):
            tile = self.generate_tile(seed=(seed + i) if seed is not None else None)
            audio = self.tile_to_audio(tile)
            audio_segments.append(audio)

        if not audio_segments:
            return np.zeros(0, dtype=np.float32)

        # Crossfade between consecutive segments
        blended = audio_segments[0]
        for next_seg in audio_segments[1:]:
            blended = self.crossfade_mixer.crossfade(blended, next_seg)

        return blended

    def stream(self, num_tiles, seed=None):
        """Yield audio segments one at a time for real-time playback.

        Parameters
        ----------
        num_tiles : int
            Number of tiles to generate.
        seed : int, optional
            Base seed for reproducibility.

        Yields
        ------
        np.ndarray
            Audio segment for each tile.
        """
        for i in range(num_tiles):
            tile = self.generate_tile(seed=(seed + i) if seed is not None else None)
            audio = self.tile_to_audio(tile)
            yield audio
