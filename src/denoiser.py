import numpy as np


class Denoiser:
    """A lightweight U-Net-like denoiser for spectrogram tiles.

    The denoiser operates on 2D spectrogram tiles of shape (n_freq, n_frames).
    It applies 1D convolutions over the frequency axis and uses residual
    connections to predict the noise added to a clean spectrogram.

    Parameters
    ----------
    n_freq : int
        Number of frequency bins in the input spectrogram.
    n_frames : int
        Number of time frames in the input spectrogram.
    hidden_channels : int
        Number of channels in the convolutional layers.
    """

    def __init__(self, n_freq=257, n_frames=128, hidden_channels=64):
        self.n_freq = n_freq
        self.n_frames = n_frames
        self.hidden_channels = hidden_channels

        # Initialize weights with small random values.
        # In a real implementation, these would be trained.
        rng = np.random.default_rng(0)
        self.weights = {}
        self.biases = {}

        # Encoder convolutions (frequency-wise)
        self.weights['enc1'] = rng.normal(0, 0.1, (hidden_channels, 1, 3))
        self.biases['enc1'] = np.zeros(hidden_channels)
        self.weights['enc2'] = rng.normal(0, 0.1, (hidden_channels, hidden_channels, 3))
        self.biases['enc2'] = np.zeros(hidden_channels)

        # Decoder convolutions
        self.weights['dec1'] = rng.normal(0, 0.1, (hidden_channels, hidden_channels, 3))
        self.biases['dec1'] = np.zeros(hidden_channels)
        self.weights['dec2'] = rng.normal(0, 0.1, (1, hidden_channels, 3))
        self.biases['dec2'] = np.zeros(1)

    def _conv1d(self, x, weight, bias):
        """Apply a 1D convolution along the frequency axis.

        Parameters
        ----------
        x : np.ndarray
            Input of shape (in_channels, n_freq, n_frames).
        weight : np.ndarray
            Weight of shape (out_channels, in_channels, kernel_size).
        bias : np.ndarray
            Bias of shape (out_channels,).

        Returns
        -------
        np.ndarray
            Output of shape (out_channels, n_freq, n_frames).
        """
        out_channels, in_channels, kernel_size = weight.shape
        n_freq, n_frames = x.shape[-2:]
        pad = kernel_size // 2
        # Zero-pad the frequency dimension
        x_pad = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='constant')
        out = np.zeros((out_channels, n_freq, n_frames), dtype=x.dtype)
        for i in range(n_freq):
            for oc in range(out_channels):
                acc = bias[oc]
                for ic in range(in_channels):
                    for k in range(kernel_size):
                        acc += weight[oc, ic, k] * x_pad[ic, i + k, :]
                out[oc, i, :] = acc
        return out

    def forward(self, x):
        """Predict the noise component given a noisy spectrogram.

        Parameters
        ----------
        x : np.ndarray
            Noisy spectrogram of shape (n_freq, n_frames).

        Returns
        -------
        np.ndarray
            Estimated noise of shape (n_freq, n_frames).
        """
        # Add channel dimension: (1, n_freq, n_frames)
        x = x[np.newaxis, :, :]

        # Encoder
        h = np.tanh(self._conv1d(x, self.weights['enc1'], self.biases['enc1']))
        h = np.tanh(self._conv1d(h, self.weights['enc2'], self.biases['enc2']))

        # Decoder
        h = np.tanh(self._conv1d(h, self.weights['dec1'], self.biases['dec1']))
        noise = self._conv1d(h, self.weights['dec2'], self.biases['dec2'])

        # Remove channel dimension
        return noise[0]

    def denoise(self, x, steps=10, noise_scale=1.0):
        """Iteratively denoise a spectrogram using the diffusion process.

        Parameters
        ----------
        x : np.ndarray
            Initial noisy spectrogram (e.g., pure noise) of shape (n_freq, n_frames).
        steps : int
            Number of diffusion steps to run.
        noise_scale : float
            Scale factor for the noise added at each step.

        Returns
        -------
        np.ndarray
            Denoised spectrogram of shape (n_freq, n_frames).
        """
        current = x.copy()
        for t in range(steps):
            # Estimate noise
            noise_pred = self.forward(current)
            # Remove a fraction of the predicted noise
            current = current - noise_scale * noise_pred / steps
        return current
