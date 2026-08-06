import numpy as np


class DiffusionScheduler:
    """Manages the noise schedule and iterative diffusion steps.

    This module provides a scheduler that controls the amount of noise
    added at each diffusion step, following a configurable noise schedule.
    It is used by the main generation loop to progressively denoise a
    spectrogram tile.
    """

    def __init__(self, noise_schedule, num_steps=50, device=None):
        """Initialize the scheduler.

        Parameters
        ----------
        noise_schedule : NoiseSchedule
            An instance of the NoiseSchedule class providing beta values.
        num_steps : int
            Number of diffusion steps.
        device : optional
            Device for computation (unused, kept for API compatibility).
        """
        self.noise_schedule = noise_schedule
        self.num_steps = num_steps
        self.device = device
        self._betas = None
        self._alphas = None
        self._alpha_cumprod = None
        self._timesteps = None
        self._step = 0

    def _prepare(self):
        """Precompute beta, alpha, and cumulative product arrays."""
        if self._betas is not None:
            return
        betas = self.noise_schedule.get_betas(self.num_steps)
        self._betas = np.asarray(betas, dtype=np.float32)
        self._alphas = 1.0 - self._betas
        self._alpha_cumprod = np.cumprod(self._alphas)
        self._timesteps = np.arange(self.num_steps)

    def reset(self):
        """Reset the scheduler to the initial state."""
        self._step = 0
        self._prepare()

    def get_betas(self):
        """Return the beta schedule array."""
        self._prepare()
        return self._betas

    def get_alphas(self):
        """Return the alpha schedule array."""
        self._prepare()
        return self._alphas

    def get_alpha_cumprod(self):
        """Return the cumulative product of alphas."""
        self._prepare()
        return self._alpha_cumprod

    def get_timesteps(self):
        """Return the timestep indices."""
        self._prepare()
        return self._timesteps

    def get_current_step(self):
        """Return the current step index."""
        return self._step

    def step(self, denoise_fn, x_t, noise, **kwargs):
        """Perform one denoising step.

        Parameters
        ----------
        denoise_fn : callable
            Function that takes (x_t, t) and returns predicted noise.
        x_t : np.ndarray
            Current noisy spectrogram tile.
        noise : np.ndarray
            Noise tensor (unused in this simple implementation).
        **kwargs : dict
            Additional keyword arguments passed to denoise_fn.

        Returns
        -------
        np.ndarray
            The denoised spectrogram tile after one step.
        """
        self._prepare()
        if self._step >= self.num_steps:
            raise RuntimeError("Scheduler has already completed all steps.")
        t = self._timesteps[self._step]
        alpha_t = self._alphas[t]
        alpha_cumprod_t = self._alpha_cumprod[t]
        # Predict the noise using the denoiser
        predicted_noise = denoise_fn(x_t, t, **kwargs)
        # Simple denoising update: x_{t-1} = (x_t - beta_t * predicted_noise) / sqrt(alpha_t)
        # This is a simplified update; a full DDPM would use more complex formulas.
        x_prev = (x_t - self._betas[t] * predicted_noise) / np.sqrt(alpha_t)
        # Add a small amount of noise for stochasticity (optional)
        if noise is not None:
            x_prev = x_prev + np.sqrt(self._betas[t]) * noise
        self._step += 1
        return x_prev

    def add_noise(self, x_0, t, noise):
        """Add noise to a clean tile at timestep t.

        Parameters
        ----------
        x_0 : np.ndarray
            Clean spectrogram tile.
        t : int
            Timestep index.
        noise : np.ndarray
            Noise to add.

        Returns
        -------
        np.ndarray
            Noisy tile.
        """
        self._prepare()
        alpha_cumprod_t = self._alpha_cumprod[t]
        sqrt_alpha_cumprod = np.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - alpha_cumprod_t)
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
}
