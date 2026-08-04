import argparse
import logging
import sys
import time

import numpy as np

from .audio_engine import AudioEngine
from .augmentation import SpectrogramAugmenter
from .config import Config
from .denoiser import Denoiser
from .scheduler import Scheduler
from .spectrogram import Spectrogram
from .spectrum_inverse import SpectrumInverse

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Music Box - AI-generated ambient music")
    parser.add_argument("-c", "--config", type=str, help="Path to config JSON file")
    parser.add_argument("-d", "--device", type=str, help="Audio output device name or index")
    parser.add_argument("-s", "--steps", type=int, help="Number of diffusion steps")
    parser.add_argument("-t", "--tempo", type=float, help="Tempo in BPM")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        config = Config(args.config)
        if args.device:
            config.data["device"] = args.device
        if args.steps:
            config.data["diffusion_steps"] = args.steps
        if args.tempo:
            config.data["tempo"] = args.tempo

        logger.info("Configuration loaded: %s", config.data)

        # Initialize components
        sample_rate = config.data["sample_rate"]
        n_freq = config.data["n_freq"]
        n_frames = config.data["n_frames"]
        hidden_channels = config.data["hidden_channels"]

        spectrogram = Spectrogram(n_freq=n_freq, n_frames=n_frames)
        inverse = SpectrumInverse(sample_rate=sample_rate, n_freq=n_freq)
        denoiser = Denoiser(
            n_freq=n_freq,
            n_frames=n_frames,
            hidden_channels=hidden_channels,
            schedule=config.data["noise_schedule"],
        )
        augmenter = SpectrogramAugmenter()
        scheduler = Scheduler(
            sample_rate=sample_rate,
            n_frames=n_frames,
            hop_length=n_frames // 2,
            crossfade_frames=8,
        )
        audio_engine = AudioEngine(
            sample_rate=sample_rate,
            block_size=config.data["block_size"],
            crossfade_samples=config.data["crossfade_samples"],
            device=config.data["device"],
        )

        # Start audio engine
        audio_engine.start()

        logger.info("Diffusion Music Box started. Press Ctrl+C to stop.")

        # Main loop: generate and play segments
        step = 0
        try:
            while True:
                # Generate a noise tile
                noise = np.random.randn(n_freq, n_frames).astype(np.float32)
                # Augment the noise for variation
                noise = augmenter.random_time_shift(noise, max_shift=8)
                noise = augmenter.random_freq_shift(noise, max_shift=4)
                noise = augmenter.random_scale(noise, scale_range=(0.9, 1.1))

                # Run diffusion denoising
                tile = denoiser.denoise(noise, steps=config.data["diffusion_steps"])

                # Schedule the tile (handles overlap/crossfade)
                segment = scheduler.schedule(tile)

                # Convert to audio and play
                audio = inverse.synthesize(segment)
                audio_engine.play(audio)

                # Wait for the segment duration
                duration = len(audio) / sample_rate
                time.sleep(duration * 0.5)  # Overlap with next generation

                step += 1
                if step % 10 == 0:
                    logger.info("Generated segment %d", step)

        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            audio_engine.stop()

        return 0

    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
