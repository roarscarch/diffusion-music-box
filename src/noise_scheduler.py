import numpy as np


class NoiseScheduler:
    """Manage the noise schedule for the diffusion process.

    This class provides a set of beta values (noise levels) for each diffusion
    step, along with derived quantities needed for the forward and reverse
    processes. It supports linear and cosine schedules, which are common in
    diffusion models.

    Parameters
    ----------
    num_steps : int
        Number of diffusion steps.
    schedule : str, optional
        Type of schedule: 'linear' or 'cosine'. Default is 'linear'.
    beta_start : float, optional
        Starting beta value for linear schedule. Default is 0.0001.
    beta_end : float, optional
        Ending beta value for linear schedule. Default is 0.02.
    """

    def __init__(self, num_steps, schedule='linear', beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps
        self.schedule = schedule.lower()
        self.beta_start = beta_start
        self.beta_end = beta_end

        if self.schedule == 'linear':
            self.betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)
        elif self.schedule == 'cosine':
            self.betas = self._cosine_betas(num_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}. Use 'linear' or 'cosine'.")

        # Precompute derived quantities
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = np.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod[:-1]) / (1.0 - self.alpha_cumprod[1:]) if num_steps > 1 else self.betas

    def _cosine_betas(self, num_steps, max_beta=0.999):
        """Generate cosine schedule betas."""
        steps = np.arange(num_steps + 1, dtype=np.float32)
        alpha_bar = np.cos((steps / num_steps + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = np.minimum(1.0 - alpha_bar[1:] / alpha_bar[:-1], max_beta)
        return betas

    def add_noise(self, x0, t, noise=None, rng=None):
        """Add noise at a given diffusion step.

        Parameters
        ----------
        x0 : np.ndarray
            Clean data (e.g., spectrogram tile).
        t : int
            Diffusion step index (0-based).
        noise : np.ndarray, optional
            Pre-generated noise. If None, generate using rng.
        rng : np.random.Generator, optional
            Random number generator. If None, use default.

        Returns
        -------
        tuple of np.ndarray
            (noisy_sample, noise) where noise is the added noise.
        """
        if noise is None:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.standard_normal(x0.shape).astype(np.float32)
        sqrt_alpha = self.sqrt_alpha_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t]
        noisy = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return noisy, noise

    def get_alpha_cumprod(self, t):
        """Return cumulative product of alphas at step t."""
        return self.alpha_cumprod[t]

    def get_beta(self, t):
        """Return beta at step t."""
        return self.betas[t]

    def get_sqrt_recip_alpha(self, t):
        """Return sqrt(1/alpha) at step t."""
        return self.sqrt_recip_alpha[t]

    def get_posterior_variance(self, t):
        """Return posterior variance at step t (for sampling)."""
        if t == 0:
            return self.betas[0]
        return self.posterior_variance[t-1]

    def __len__(self):
        return self.num_steps

    def __repr__(self):
        return f"NoiseScheduler(num_steps={self.num_steps}, schedule={self.schedule!r}, beta_start={self.beta_start}, beta_end={self.beta_end})"
