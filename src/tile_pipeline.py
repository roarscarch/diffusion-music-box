import numpy as np
from src.spectrogram import compute_spectrogram, spectrogram_to_audio
from src.spectrum_inverse import invert_spectrogram
from src.normalization import normalize_spectrogram, denormalize_spectrogram
from src.overlap_add import overlap_add
from src.augmentation import SpectrogramAugmenter


class TilePipeline:
    """Orchestrates the generation of a seamless audio segment from a spectrogram tile.

    This pipeline ties together the core modules: it takes a raw spectrogram tile
    (or generates one from noise), normalizes it, applies optional augmentation,
    converts it back to audio using the inverse transform, and finally applies
    overlap-add crossfading to ensure smooth transitions between segments.

    The pipeline is designed to be used in a loop to produce endless ambient music.
    """

    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        freq_bins=128,
        time_frames=128,
        augmenter=None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        self.augmenter = augmenter if augmenter is not None else SpectrogramAugmenter()

    def generate_audio_from_spectrogram(self, spectrogram):
        """Convert a 2D spectrogram tile into audio samples.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames) representing magnitude
            spectrogram (or log-magnitude).

        Returns
        -------
        np.ndarray
            1D audio samples as float32.
        """
        # Ensure spectrogram is 2D and has expected shape
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")
        if spectrogram.shape[0] != self.freq_bins or spectrogram.shape[1] != self.time_frames:
            raise ValueError(
                f"Spectrogram shape {spectrogram.shape} does not match expected ({self.freq_bins}, {self.time_frames})"
            )

        # Apply augmentation for variety (e.g., time/freq shift, scale)
        augmented = self.augmenter.random_time_shift(spectrogram)
        augmented = self.augmenter.random_freq_shift(augmented)
        augmented = self.augmenter.random_scale(augmented)

        # Convert magnitude spectrogram to audio using inverse STFT
        # This assumes the spectrogram is magnitude (not complex)
        audio = invert_spectrogram(
            augmented,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
        )

        # Normalize audio to avoid clipping
        peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
        if peak > 0:
            audio = audio / peak * 0.9

        return audio.astype(np.float32)

    def generate_segment_from_noise(self, denoiser, steps=20, noise_schedule=None):
        """Generate a new audio segment by denoising a random noise tile.

        This method runs the diffusion process: starting from random noise,
        iteratively apply the denoiser to produce a clean spectrogram tile,
        then convert to audio.

        Parameters
        ----------
        denoiser : object
            A denoiser object with a `denoise` method that accepts a noisy
            spectrogram and a noise level, returning a denoised version.
        steps : int, optional
            Number of diffusion steps to run.
        noise_schedule : object, optional
            Object with `get_noise_level(step, total_steps)` method.

        Returns
        -------
        np.ndarray
            1D audio samples.
        """
        rng = np.random.default_rng()
        # Start with random noise in spectrogram domain
        spectrogram = rng.standard_normal((self.freq_bins, self.time_frames)).astype(np.float32)

        if noise_schedule is None:
            # Default: linear schedule from 1.0 to 0.0
            def default_schedule(step, total):
                return 1.0 - step / max(total - 1, 1)
            noise_schedule = type("SimpleSchedule", (), {"get_noise_level": staticmethod(default_schedule)})()

        for step in range(steps):
            noise_level = noise_schedule.get_noise_level(step, steps)
            spectrogram = denoiser.denoise(spectrogram, noise_level)

        return self.generate_audio_from_spectrogram(spectrogram)

    def seamless_loop(self, segments, crossfade_samples=256):
        """Stitch audio segments together with crossfades to create a seamless loop.

        Parameters
        ----------
        segments : list of np.ndarray
            List of 1D audio arrays to concatenate.
        crossfade_samples : int
            Number of samples for crossfade between consecutive segments.

        Returns
        -------
        np.ndarray
            Combined audio array.
        """
        return overlap_add(segments, crossfade_samples)
