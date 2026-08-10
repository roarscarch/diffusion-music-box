import numpy as np


class NoiseSchedule:
    """Generate noise schedules for iterative diffusion.

    This module provides different noise schedules (e.g., linear, cosine,
    quadratic) that control the amount of noise added at each diffusion step.
    It is used by the denoiser to gradually denoise a spectrogram tile from
    pure noise to a coherent audio structure.

    Parameters
    ----------
    schedule_type : str, optional
        Type of schedule: 'linear', 'cosine', 'quadratic', or 'sqrt'.
    total_steps : int, optional
        Number of diffusion steps. Default is 10.
    beta_start : float, optional
        Starting noise level for linear schedule. Default is 0.0001.
    beta_end : float, optional
        Ending noise level for linear schedule. Default is 0.02.
    """

    def __init__(self, schedule_type='linear', total_steps=10,
                 beta_start=0.0001, beta_end=0.02):
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._betas = None
        self._alphas = None
        self._alpha_cumprod = None
        self._compute()

    def _compute(self):
        """Compute the noise schedule arrays.

        Stores beta, alpha, and cumulative alpha (alpha_cumprod) arrays.
        The schedule defines the variance of the noise added at each step.
        """
        steps = self.total_steps
        if self.schedule_type == 'linear':
            betas = np.linspace(self.beta_start, self.beta_end, steps)
        elif self.schedule_type == 'cosine':
            # Cosine schedule as per improved diffusion (Nichol & Dhariwal)
            s = 0.008
            t = np.arange(steps + 1, dtype=np.float64) / steps
            f_t = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = f_t / f_t[0]
            betas = np.clip(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], 0, 0.999)
        elif self.schedule_type == 'quadratic':
            betas = np.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, steps) ** 2
        elif self.schedule_type == 'sqrt':
            betas = np.sqrt(np.linspace(self.beta_start, self.beta_end, steps))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        self._betas = betas.astype(np.float32)
        self._alphas = 1.0 - self._betas
        self._alpha_cumprod = np.cumprod(self._alphas)

    @property
    def betas(self):
        """Noise variance at each step (array of length total_steps)."""
        return self._betas

    @property
    def alphas(self):
        """1 - beta at each step."""
        return self._alphas

    @property
    def alpha_cumprod(self):
        """Cumulative product of alphas."""
        return self._alpha_cumprod

    def get_noise_level(self, step):
        """Return the noise level (variance) for a given step.

        Parameters
        ----------
        step : int
            Diffusion step index (0 to total_steps-1).

        Returns
        -------
        float
            Noise variance for that step.
        """
        if step < 0 or step >= self.total_steps:
            raise IndexError(f"Step {step} out of range [0, {self.total_steps})")
        return float(self._betas[step])

    def get_signal_rate(self, step):
        """Return the signal retention rate (sqrt of alpha_cumprod).

        This is used for adding noise during the forward process: the signal
        is scaled by sqrt(alpha_cumprod) and noise by sqrt(1 - alpha_cumprod).
        """
        return float(np.sqrt(self._alpha_cumprod[step]))

    def add_noise(self, x, step, rng=None):
        """Add noise to a clean signal according to the schedule.

        This simulates the forward diffusion process. Given a clean tile x,
        it returns a noisy version after `step` steps.

        Parameters
        ----------
        x : np.ndarray
            Clean signal (spectrogram tile).
        step : int
            Number of steps to forward diffuse (0 to total_steps-1).
        rng : np.random.Generator, optional
            Random generator for reproducibility.

        Returns
        -------
        tuple of (noisy, noise)
            The noisy signal and the added noise.
        """
        if rng is None:
            rng = np.random.default_rng()
        alpha_cumprod = self._alpha_cumprod[step]
        signal_scale = np.sqrt(alpha_cumprod)
        noise_scale = np.sqrt(1 - alpha_cumprod)
        noise = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
        noisy = signal_scale * x + noise_scale * noise
        return noisy, noise

    def get_timesteps(self):
        """Return an array of step indices from most noise to least."""
        return np.arange(self.total_steps - 1, -1, -1)
