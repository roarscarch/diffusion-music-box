import argparse
import logging
import sys
import time
import threading

import numpy as np

from .audio_buffer import AudioSegmentBuffer
from .audio_engine import AudioEngine
from .augmentation import SpectrogramAugmenter
from .config import Config
from .denoiser import Denoiser
from .keyboard_controller import KeyboardController
from .midi_controller import MIDIController
from .noise_schedule import NoiseSchedule
from .normalization import normalize_spectrogram, denormalize_spectrogram
from .overlap_add import overlap_add
from .pitch_shift import pitch_shift_tile
from .scheduler import DiffusionScheduler
from .spectrogram import SpectrogramProcessor
from .spectrum_inverse import inverse_spectrogram
from .tempo_scheduler import TempoScheduler

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Music Box - AI-generated ambient music from noise")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--device", type=str, help="Output device name or index")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    # Load config
    config = Config()
    if args.config:
        config.load(args.config)

    # Override with CLI args
    if args.sample_rate:
        config.sample_rate = args.sample_rate
    if args.block_size:
        config.block_size = args.block_size
    if args.device:
        config.device = args.device

    # Initialize components
    spectrogram_processor = SpectrogramProcessor(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_freq_bins=config.n_freq_bins,
    )
    denoiser = Denoiser(
        n_freq_bins=config.n_freq_bins,
        n_time_frames=config.n_time_frames,
    )
    noise_schedule = NoiseSchedule(
        schedule_type=config.noise_schedule_type,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    scheduler = DiffusionScheduler(
        denoiser=denoiser,
        noise_schedule=noise_schedule,
    )
    augmenter = SpectrogramAugmenter(seed=config.seed)
    tempo_scheduler = TempoScheduler(initial_bpm=config.tempo)
    audio_buffer = AudioSegmentBuffer(max_segments=config.max_buffered_segments)
    audio_engine = AudioEngine(
        sample_rate=config.sample_rate,
        block_size=config.block_size,
        crossfade_samples=config.crossfade_samples,
        device=config.device,
    )

    # Set up controllers
    keyboard_controller = KeyboardController()
    midi_controller = MIDIController()

    # Generation parameters
    params = {
        "diffusion_steps": config.diffusion_steps,
        "noise_level": config.noise_level,
        "tempo": config.tempo,
    }

    def update_params():
        """Update parameters from controllers and tempo scheduler."""
        # From keyboard
        key_state = keyboard_controller.get_state()
        if key_state:
            if "+" in key_state:
                params["diffusion_steps"] = min(100, params["diffusion_steps"] + 1)
            if "-" in key_state:
                params["diffusion_steps"] = max(1, params["diffusion_steps"] - 1)
            if "n" in key_state:
                params["noise_level"] = min(1.0, params["noise_level"] + 0.05)
            if "m" in key_state:
                params["noise_level"] = max(0.0, params["noise_level"] - 0.05)
            if "t" in key_state:
                params["tempo"] = min(200, params["tempo"] + 1)
            if "g" in key_state:
                params["tempo"] = max(20, params["tempo"] - 1)

        # From MIDI
        midi_state = midi_controller.get_state()
        if midi_state:
            # Simple mapping: note 0-10 controls steps, 11-20 controls noise, 21+ controls tempo
            for note, velocity in midi_state.items():
                if note < 11:
                    params["diffusion_steps"] = int(velocity / 127 * 100)
                elif note < 22:
                    params["noise_level"] = velocity / 127
                else:
                    params["tempo"] = int(velocity / 127 * 200)

        # Tempo scheduler
        params["tempo"] = tempo_scheduler.get_current_bpm()

    def generate_segment():
        """Generate one audio segment and return it."""
        # Update tempo and other params
        update_params()

        # Create noise spectrogram tile
        noise = np.random.randn(config.n_freq_bins, config.n_time_frames).astype(np.float32)
        # Apply augmentations
        noise = augmenter.random_time_shift(noise)
        noise = augmenter.random_freq_shift(noise)
        noise = augmenter.random_scale(noise)

        # Run diffusion
        denoised = scheduler.run(noise, steps=params["diffusion_steps"], noise_level=params["noise_level"])

        # Normalize
        denoised, norm_params = normalize_spectrogram(denoised)

        # Pitch shift for variation
        denoised = pitch_shift_tile(denoised, shift_semitones=0.0)

        # Inverse spectrogram to audio
        audio = inverse_spectrogram(
            denoised,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            sample_rate=config.sample_rate,
        )

        # Denormalize (audio is already in time domain, but just in case)
        # Actually, we need to apply inverse normalization? For now just return
        return audio

    def generation_loop():
        """Background thread that generates segments and adds to buffer."""
        logger.info("Starting generation loop")
        while not stop_event.is_set():
            try:
                segment = generate_segment()
                audio_buffer.add(segment)
                logger.debug(f"Generated segment of length {len(segment)}")
            except Exception as e:
                logger.error(f"Generation error: {e}