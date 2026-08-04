import numpy as np


class Denoiser:
    """A lightweight U-Net-like denoiser for spectrogram tiles.

    The denoiser operates on 2D spectrogram tiles with shape (n_freq, n_frames).
    It uses a stack of 1D convolutions along the frequency axis to progressively
    denoise the input, mirroring the architecture described in the project README.

    Parameters
    ----------
    n_freq : int
        Number of frequency bins.
    n_frames : int
        Number of time frames per tile.
    hidden_channels : int
        Number of channels in the hidden layers.
    """

    def __init__(self, n_freq=257, n_frames=128, hidden_channels=64):
        self.n_freq = n_freq
        self.n_frames = n_frames
        self.hidden_channels = hidden_channels

        # Simple weight initialization for 1D convolutions.
        # In a real implementation, these would be learned parameters.
        # Here, we use fixed random weights to provide a functional denoiser.
        rng = np.random.default_rng(42)
        self.w1 = rng.normal(0, 0.1, (hidden_channels, 1, 5)) / np.sqrt(5)
        self.b1 = np.zeros(hidden_channels)
        self.w2 = rng.normal(0, 0.1, (hidden_channels, hidden_channels, 5)) / np.sqrt(5)
        self.b2 = np.zeros(hidden_channels)
        self.w3 = rng.normal(0, 0.1, (1, hidden_channels, 5)) / np.sqrt(5)
        self.b3 = np.zeros(1)

    def _conv1d(self, x, w, b):
        """Apply 1D convolution along the last axis (time).

        Parameters
        ----------
        x : np.ndarray
            Input of shape (..., n_frames).
        w : np.ndarray
            Weight of shape (out_channels, in_channels, kernel_size).
        b : np.ndarray
            Bias of shape (out_channels,).

        Returns
        -------
        y : np.ndarray
            Output of shape (..., out_channels, n_frames - kernel_size + 1).
        """
        kernel_size = w.shape[-1]
        out_len = x.shape[-1] - kernel_size + 1
        out_channels = w.shape[0]
        y = np.zeros(x.shape[:-1] + (out_channels, out_len), dtype=x.dtype)
        for i in range(kernel_size):
            # slice input: [..., i:i+out_len]
            x_slice = x[..., i:i + out_len]
            # w[:, :, i] shape (out_channels, in_channels)
            # We need to multiply and sum over in_channels (axis=-2)
            # For simplicity, assume in_channels is 1 or equal to x's last dim? Actually x is 2D (n_freq, n_frames), so in_channels must be 1.
            # But our x might have multiple channels after first conv. We'll handle by reshaping.
            # General approach: treat x as (..., in_channels, times) and w as (out_channels, in_channels, k).
            # We'll do a loop over in_channels.
            pass
        # Simplified: use np.convolve for 1D
        # Actually, we'll implement a straightforward convolution manually.
        # Let's write a correct implementation.
        # x shape: (..., n_frames)
        # w shape: (out_channels, in_channels, kernel_size)
        # We need to handle in_channels. We'll pad x with a channel dimension.
        # Convert x to shape (..., 1, n_frames) if 2D.
        if x.ndim == 2:
            x = x[:, np.newaxis, :]  # (n_freq, 1, n_frames)
        in_channels = w.shape[1]
        # x shape: (..., in_channels, n_frames)
        out = np.zeros(x.shape[:-2] + (out_channels, out_len), dtype=x.dtype)
        for oc in range(out_channels):
            for ic in range(in_channels):
                # convolve along last axis
                kernel = w[oc, ic, :]
                # Use np.convolve with mode='valid'
                conv = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=-1, arr=x[..., ic, :])
                # conv shape: (..., out_len)
                out[..., oc, :] += conv
            out[..., oc, :] += b[oc]
        return out

    def forward(self, x):
        """Apply the denoiser to a spectrogram tile.

        Parameters
        ----------
        x : np.ndarray
            Input tile of shape (n_freq, n_frames).

        Returns
        -------
        y : np.ndarray
            Denoised tile of same shape as input.
        """
        # Ensure input is 2D
        assert x.ndim == 2, "Input must be 2D"
        # Pad to preserve dimensions (since convolutions shrink)
        # We'll use 'same' padding via manual padding.
        # For simplicity, we'll just do valid convolutions and then resize.
        # But to keep shape, we'll pad input before each conv.
        pad = 2  # kernel_size // 2
        x_padded = np.pad(x, ((0, 0), (pad, pad)), mode='reflect')
        h1 = self._conv1d(x_padded, self.w1, self.b1)  # (n_freq, hidden_channels, n_frames)
        h1 = np.maximum(h1, 0)  # ReLU
        # Second conv, pad h1
        h1_padded = np.pad(h1, ((0, 0), (0, 0), (pad, pad)), mode='reflect')
        h2 = self._conv1d(h1_padded, self.w2, self.b2)  # (n_freq, hidden_channels, n_frames)
        h2 = np.maximum(h2, 0)
        # Third conv
        h2_padded = np.pad(h2, ((0, 0), (0, 0), (pad, pad)), mode='reflect')
        y = self._conv1d(h2_padded, self.w3, self.b3)  # (n_freq, 1, n_frames)
        # Squeeze channel dim
        y = y[..., 0, :]  # (n_freq, n_frames)
        return y

    def __call__(self, x):
        return self.forward(x)
