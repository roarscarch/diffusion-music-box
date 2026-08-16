import numpy as np


class NoiseSchedule:
    """Generate noise schedules for diffusion models.

    This module provides functions to create beta schedules (noise levels)
    for the forward diffusion process. It supports linear, cosine, and
    quadratic schedules, which are commonly used in diffusion models to
    control how noise is added over time steps.

    The schedule is an array of beta values, where each beta represents the
    variance of the noise added at a particular diffusion step. These betas
    are used in the forward process to progressively corrupt data, and in the
    reverse process to guide denoising.

    Parameters
    ----------
    num_steps : int
        Total number of diffusion steps.
    schedule_type : str, optional
        Type of schedule: 'linear', 'cosine', or 'quadratic'.
    beta_start : float, optional
        Starting beta value (for linear/quadratic schedules).
    beta_end : float, optional
        Ending beta value (for linear/quadratic schedules).
    """

    def __init__(self, num_steps=100, schedule_type='linear', beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self._generate_betas()

    def _generate_betas(self):
        """Generate the beta schedule array."""
        if self.schedule_type == 'linear':
            return np.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=np.float32)
        elif self.schedule_type == 'cosine':
            return self._cosine_betas()
        elif self.schedule_type == 'quadratic':
            return self._quadratic_betas()
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def _cosine_betas(self):
        """Generate betas using a cosine schedule (as in Nichol & Dhariwal)."""
        steps = np.arange(self.num_steps + 1, dtype=np.float32)
        alpha_bar = np.cos((steps / self.num_steps + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return np.clip(betas, 0.0, 0.999).astype(np.float32)

    def _quadratic_betas(self):
        """Generate betas with a quadratic interpolation between start and end."""
        steps = np.linspace(0, 1, self.num_steps, dtype=np.float32)
        betas = self.beta_start + (self.beta_end - self.beta_start) * steps ** 2
        return betas.astype(np.float32)

    def get_alpha(self):
        """Compute alpha values (1 - beta) for the schedule."""
        return 1.0 - self.betas

    def get_alpha_bar(self):
        """Compute cumulative product of alphas."""
        alphas = self.get_alpha()
        alpha_bar = np.cumprod(alphas, dtype=np.float32)
        return alpha_bar

    def __len__(self):
        return self.num_steps

    def __repr__(self):
        return (f"NoiseSchedule(num_steps={self.num_steps}, "
                f"schedule_type='{self.schedule_type}')")
