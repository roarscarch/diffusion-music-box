import numpy as np


class NoiseSchedule:
    """Manage noise levels for the diffusion process.

    This class provides linear and cosine beta schedules, which determine
    how much noise is added at each diffusion step. The schedule can be
    used to compute alpha and alpha_bar values for forward and reverse
    diffusion processes.
    """

    def __init__(self, steps=100, schedule_type='linear', beta_start=0.0001, beta_end=0.02):
        """
        Parameters
        ----------
        steps : int
            Number of diffusion steps.
        schedule_type : str
            Type of schedule: 'linear' or 'cosine'.
        beta_start : float
            Starting beta value for linear schedule.
        beta_end : float
            Ending beta value for linear schedule.
        """
        self.steps = steps
        self.schedule_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def _compute_betas(self):
        """Compute beta values for the chosen schedule."""
        if self.schedule_type == 'linear':
            return np.linspace(self.beta_start, self.beta_end, self.steps)
        elif self.schedule_type == 'cosine':
            return self._cosine_betas()
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def _cosine_betas(self):
        """Cosine schedule from 'Improved Denoising Diffusion Probabilistic Models'."""
        steps = self.steps
        s = 0.008
        x = np.linspace(0, steps, steps + 1) / steps
        alpha_bar = np.cos((x + s) / (1 + s) * (np.pi / 2)) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return np.clip(betas, 0.0, 0.999)

    def get_alpha_bar(self, t):
        """Return alpha_bar at step t (0-indexed)."""
        return self.alpha_bars[t]

    def get_alpha(self, t):
        """Return alpha at step t (0-indexed)."""
        return self.alphas[t]

    def get_beta(self, t):
        """Return beta at step t (0-indexed)."""
        return self.betas[t]

    def add_noise(self, x0, t, noise=None):
        """Add noise according to forward diffusion process.

        Parameters
        ----------
        x0 : np.ndarray
            Clean data (e.g., spectrogram tile).
        t : int
            Diffusion step index (0..steps-1).
        noise : np.ndarray, optional
            Noise to add; if None, generated randomly.

        Returns
        -------
        np.ndarray
            Noisy sample at step t.
        """
        if noise is None:
            noise = np.random.randn(*x0.shape).astype(np.float32)
        alpha_bar = self.alpha_bars[t]
        return np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise

    def __len__(self):
        return self.steps
