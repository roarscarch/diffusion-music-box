import numpy as np


class Denoiser:
    """Iterative diffusion denoiser for spectrogram tiles.

    This module implements a lightweight diffusion process on 2D frequency-time
    tiles. The denoiser applies a series of steps that progressively remove
    noise from a spectrogram, using a simple spectral gating mechanism combined
    with a learned noise schedule. The result is a smooth, evolving ambient
    texture that can be converted back to audio.

    Parameters
    ----------
    noise_schedule : Scheduler
        Scheduler that defines the noise levels for each diffusion step.
    steps : int, optional
        Number of diffusion steps to run per generation call. Defaults to 50.
    """

    def __init__(self, noise_schedule, steps=50):
        self.noise_schedule = noise_schedule
        self.steps = steps

    def denoise(self, tile, current_step=None):
        """Run one denoising step on a spectrogram tile.

        Parameters
        ----------
        tile : np.ndarray
            2D array representing a spectrogram tile (frequency x time).
        current_step : int, optional
            The current diffusion step index. If None, uses the internal step counter.

        Returns
        -------
        np.ndarray
            Denoised tile.
        """
        if current_step is None:
            current_step = self.steps - 1
        # Compute noise level for this step
        noise_level = self.noise_schedule.get_noise(current_step)
        # Simple spectral gating: keep only magnitudes above a threshold
        # that depends on the noise level
        threshold = noise_level * 0.5
        # Apply soft thresholding (shrinkage) to the magnitude spectrum
        # This is a placeholder for a more sophisticated denoiser
        magnitude = np.abs(tile)
        phase = np.angle(tile)
        # Soft threshold
        magnitude_denoised = np.maximum(magnitude - threshold, 0)
        # Reconstruct with original phase
        tile_denoised = magnitude_denoised * np.exp(1j * phase)
        return tile_denoised

    def generate(self, tile_shape, rng=None):
        """Generate a new spectrogram tile by iteratively denoising pure noise.

        Parameters
        ----------
        tile_shape : tuple of int
            Shape of the tile (freq_bins, time_frames).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray
            Generated spectrogram tile (complex-valued).
        """
        if rng is None:
            rng = np.random.default_rng()
        # Start from complex noise
        noise = rng.standard_normal(tile_shape) + 1j * rng.standard_normal(tile_shape)
        tile = noise
        for step in reversed(range(self.steps)):
            tile = self.denoise(tile, current_step=step)
            # Add a small amount of noise back to keep the process stochastic
            noise_level = self.noise_schedule.get_noise(step)
            if noise_level > 0:
                tile = tile + noise_level * (rng.standard_normal(tile_shape) + 1j * rng.standard_normal(tile_shape))
        return tile
