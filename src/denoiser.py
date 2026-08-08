import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Denoiser(nn.Module):
    """A lightweight U-Net-like denoiser for 2D spectrogram tiles.

    The model operates on a fixed-size 2D frequency-time image and
    predicts the noise component. It uses 1D convolutions over the
    frequency axis with skip connections to preserve high-frequency
    details. The architecture is intentionally small to run in real-time
    on CPU.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 1 for magnitude spectrogram).
    base_channels : int
        Number of channels in the first convolutional layer.
    depth : int
        Number of downsampling/upsampling levels.
    """

    def __init__(self, in_channels=1, base_channels=32, depth=3):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels

        # Encoder: downsampling path
        self.enc_convs = nn.ModuleList()
        self.enc_pools = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.enc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            self.enc_pools.append(nn.MaxPool2d(2))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * 2, in_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(inplace=True),
        )
        in_ch = in_ch * 2

        # Decoder: upsampling path with skip connections
        self.dec_convs = nn.ModuleList()
        self.dec_ups = nn.ModuleList()
        for i in reversed(range(depth)):
            out_ch = base_channels * (2 ** i)
            self.dec_ups.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.dec_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch

        # Output layer
        self.out_conv = nn.Conv2d(in_ch, in_channels, kernel_size=1)

    def forward(self, x):
        # x: (batch, channels, freq, time)
        skips = []
        for conv, pool in zip(self.enc_convs, self.enc_pools):
            x = conv(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.dec_ups, self.dec_convs, reversed(skips)):
            x = up(x)
            # Handle size mismatch due to odd dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return self.out_conv(x)


class DiffusionStepScheduler:
    """Schedules noise levels for iterative diffusion.

    Implements a linear noise schedule from high to low noise over a
    given number of steps. The schedule can be adjusted dynamically for
    interactive control.

    Parameters
    ----------
    num_steps : int
        Number of diffusion steps.
    beta_start : float
        Initial noise level (0 to 1).
    beta_end : float
        Final noise level (0 to 1).
    """

    def __init__(self, num_steps, beta_start=0.9, beta_end=0.1):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._betas = np.linspace(beta_start, beta_end, num_steps)

    def get_beta(self, step):
        """Return the noise level for a given step."""
        step = min(max(step, 0), self.num_steps - 1)
        return float(self._betas[step])

    def set_num_steps(self, num_steps):
        """Update the number of steps and recompute the schedule."""
        self.num_steps = max(num_steps, 1)
        self._betas = np.linspace(self.beta_start, self.beta_end, self.num_steps)

    def set_beta_range(self, beta_start, beta_end):
        """Update the beta range and recompute the schedule."""
        self.beta_start = max(0.0, min(beta_start, 1.0))
        self.beta_end = max(0.0, min(beta_end, 1.0))
        self._betas = np.linspace(self.beta_start, self.beta_end, self.num_steps)

    def get_betas(self):
        """Return the full beta schedule."""
        return self._betas.copy()
