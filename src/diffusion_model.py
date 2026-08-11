import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (batch,) tensor of timesteps (float or int)
        Returns:
            (batch, dim) tensor of embeddings
        """
        half_dim = self.dim // 2
        exponents = -np.log(10000.0) * torch.arange(half_dim, device=t.device) / half_dim
        omega = t.unsqueeze(-1).float() * torch.exp(exponents).unsqueeze(0)
        embedding = torch.cat([torch.sin(omega), torch.cos(omega)], dim=-1)
        return embedding


class ResidualBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """Downsampling block."""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        return self.pool(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        # Handle size mismatch if any
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


class UNetDenoiser(nn.Module):
    """Lightweight U-Net denoiser for 2D spectrogram tiles.

    The model takes a noisy spectrogram tile (batch, 1, freq, time) and a
    diffusion timestep, and predicts the noise that was added. This is the
    core of the diffusion model.
    """

    def __init__(self, channels=(16, 32, 64), time_emb_dim=32, input_channels=1):
        super().__init__()
        self.time_embedding = SinusoidalEmbedding(time_emb_dim)

        # Initial conv that expands channels
        self.init_conv = nn.Conv2d(input_channels, channels[0], 3, padding=1)

        # Down blocks
        self.downs = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.downs.append(DownBlock(in_ch, out_ch, time_emb_dim))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = ResidualBlock(in_ch, in_ch, time_emb_dim)

        # Up blocks
        self.ups = nn.ModuleList()
        # Reverse channels for up path, concatenating with skip connections
        # The up block receives [prev_out, skip] as input channels
        for i in range(len(channels) - 1, 0, -1):
            out_ch = channels[i - 1]
            in_ch = channels[i] + (channels[i - 1] if i < len(channels) else 0)
            # Actually, the up block concatenates the upsampled input with the skip
            # So input channels = channels[i] + channels[i-1]
            self.ups.append(UpBlock(channels[i] + channels[i - 1], out_ch, time_emb_dim))

        # Final conv to 1 channel
        self.final_conv = nn.Conv2d(channels[0], input_channels, 3, padding=1)

    def forward(self, x, t):
        """
        Args:
            x: (batch, 1, freq, time) noisy spectrogram tile
            t: (batch,) timesteps (0 to 1, float)
        Returns:
            (batch, 1, freq, time) predicted noise
        """
        t_emb = self.time_embedding(t)

        h = F.silu(self.init_conv(x))
        skips = []
        for down in self.downs:
            h = down(h, t_emb)
            skips.append(h)

        h = self.bottleneck(h, t_emb)

        # Reverse skips for up path
        skips = skips[::-1]
        for up, skip in zip(self.ups, skips):
            h = up(h, skip, t_emb)

        return self.final_conv(h)


class DiffusionModel:
    """Wrapper for the diffusion process: forward noising and reverse denoising.

    This class provides methods to add noise to a spectrogram tile and to
    iteratively denoise it using the UNet. The noise schedule is linear
    between small and large noise levels.

    Parameters
    ----------
    model : nn.Module, optional
        The denoiser network. If None, a default UNetDenoiser is created.
    num_timesteps : int
        Number of diffusion steps for the reverse process.
    beta_start : float
        Starting noise level (variance).
    beta_end : float
        Ending noise level (variance).
    device : str
        Device to run on ('cpu' or 'cuda').
    """

    def __init__(self, model=None, num_timesteps=50, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.device = device
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        if model is None:
            model = UNetDenoiser()
        self.model = model.to(device)

    def _timestep_to_tensor(self, t):
        """Convert timestep indices to a float tensor in [0, 1]."""
        t = torch.as_tensor(t, dtype=torch.float