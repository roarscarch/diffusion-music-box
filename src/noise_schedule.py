import numpy as np


class NoiseSchedule:
    """Define and manage noise schedules for the diffusion process.

    The noise schedule determines how much noise is added or removed at each
    diffusion step. It is a key parameter for controlling the generation
    process: a steeper schedule leads to more drastic changes, while a
    gentler schedule produces subtler evolution. This module provides
    several built-in schedule types and allows custom schedules.

    Parameters
    ----------
    schedule_type : str, optional
        Type of schedule: 'linear', 'cosine', 'quadratic', or 'custom'.
        Default is 'cosine'.
    num_steps : int, optional
        Number of diffusion steps. Default is 50.
    beta_start : float, optional
        Starting noise level (applies to linear and quadratic schedules).
        Must be between 0 and 1. Default is 0.0001.
    beta_end : float, optional
        Ending noise level (applies to linear and quadratic schedules).
        Must be greater than beta_start. Default is 0.02.
    custom_betas : array_like, optional
        Explicit array of beta values for 'custom' schedule. Must have
        length equal to num_steps and values in [0, 1].

    Attributes
    ----------
    betas : np.ndarray
        1D float array of shape (num_steps,) containing the noise level
        for each diffusion step.
    alphas : np.ndarray
        1D float array where alphas[i] = 1 - betas[i].
    alpha_cumprod : np.ndarray
        Cumulative product of alphas, used for noising in forward process.
    """

    def __init__(self, schedule_type='cosine', num_steps=50, beta_start=0.0001,
                 beta_end=0.02, custom_betas=None):
        self.schedule_type = schedule_type
        self.num_steps = int(num_steps)
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")

        if schedule_type == 'custom':
            if custom_betas is None:
                raise ValueError("custom_betas must be provided for 'custom' schedule")
            betas = np.asarray(custom_betas, dtype=np.float32)
            if betas.shape[0] != self.num_steps:
                raise ValueError(f"custom_betas length {betas.shape[0]} does not match num_steps {self.num_steps}")
            if np.any(betas < 0) or np.any(betas > 1):
                raise ValueError("custom_betas values must be in [0, 1]")
        else:
            if not (0 <= beta_start < beta_end <= 1):
                raise ValueError("beta_start must be in [0,1] and beta_end > beta_start")
            if schedule_type == 'linear':
                betas = np.linspace(beta_start, beta_end, self.num_steps, dtype=np.float32)
            elif schedule_type == 'quadratic':
                t = np.linspace(0, 1, self.num_steps, dtype=np.float32)
                betas = beta_start + (beta_end - beta_start) * t**2
            elif schedule_type == 'cosine':
                # Cosine schedule from 'Improved Denoising Diffusion Probabilistic Models'
                # using alpha_bar = cos((t/T + s) / (1+s) * pi/2)^2
                s = 0.008
                t = np.linspace(0, 1, self.num_steps + 1, dtype=np.float32)
                alpha_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
                # Clip to avoid numerical issues
                alpha_bar = np.clip(alpha_bar, 0.0, 1.0)
                # Compute betas as 1 - alpha_bar[t+1] / alpha_bar[t]
                betas = np.zeros(self.num_steps, dtype=np.float32)
                for i in range(self.num_steps):
                    alpha_t = alpha_bar[i + 1] / alpha_bar[i]
                    betas[i] = np.clip(1.0 - alpha_t, 0.0, 1.0)
            else:
                raise ValueError(f"Unknown schedule_type: {schedule_type}")

        self.betas = betas.astype(np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas, dtype=np.float32)

    def get_beta_at_step(self, step):
        """Return the beta value for a given step index.

        Parameters
        ----------
        step : int
            Diffusion step index (0-based).

        Returns
        -------
        float
            Beta value for that step.
        """
        if step < 0 or step >= self.num_steps:
            raise IndexError(f"step {step} out of range [0, {self.num_steps})")
        return float(self.betas[step])

    def get_alpha_cumprod_at_step(self, step):
        """Return the cumulative product of alphas up to a given step.

        Parameters
        ----------
        step : int
            Diffusion step index (0-based).

        Returns
        -------
        float
            Cumulative product of alphas.
        """
        if step < 0 or step >= self.num_steps:
            raise IndexError(f"step {step} out of range [0, {self.num_steps})")
        return float(self.alpha_cumprod[step])

    def add_noise(self, x0, step, noise=None, rng=None):
        """Add noise to a clean signal according to the schedule at a given step.

        This implements the forward diffusion process: given a clean input x0,
        returns a noisy version at the specified step.

        Parameters
        ----------
        x0 : np.ndarray
            Clean input (e.g., spectrogram tile).
        step : int
            Diffusion step index (0-based).
        noise : np.ndarray, optional
            Pre-generated noise. If None, generated using rng or default.
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray
            Noisy version of x0.
        """
        if rng is None:
            rng = np.random.default_rng()
        if noise is None:
            noise = rng.standard_normal(x0.shape).astype(np.float32)

        alpha_cumprod = self.alpha_cumprod[step]
        sqrt_alpha_bar = np.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_bar = np.sqrt(1.0 - alpha_cumprod)

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def to_dict(self):
        """Return a dictionary representation of the schedule config."""
        return {
            'schedule_type': self.schedule_type,
            'num_steps': self.num_steps,
            'betas': self.betas.tolist(),
            'alphas': self.alphas.tolist(),
            'alpha_cumprod': self.alpha_cumprod.tolist(),
        }