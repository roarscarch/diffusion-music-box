import argparse
import sys
import threading
import time

import numpy as np

from .audio_engine import AudioEngine
from .config import Config
from .denoiser import Denoiser
from .generator import Generator
from .keyboard_controller import KeyboardController
from .midi_controller import MidiController
from .spectrogram import Spectrogram
from .tile_pipeline import TilePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Music Box - AI-generated ambient music")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--block-size", type=int, default=1024, help="Audio block size")
    parser.add_argument("--device", type=str, default=None, help="Output device name or index")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--no-keyboard", action="store_true", help="Disable keyboard control")
    parser.add_argument("--no-midi", action="store_true", help="Disable MIDI control")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.load(args.config) if args.config else Config()

    # Override config with CLI arguments if provided
    if args.sample_rate:
        config.sample_rate = args.sample_rate
    if args.block_size:
        config.block_size = args.block_size
    if args.device:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    # Initialize components
    rng = np.random.default_rng(config.seed)
    denoiser = Denoiser(config)
    generator = Generator(config, denoiser, rng)
    audio_engine = AudioEngine(
        sample_rate=config.sample_rate,
        block_size=config.block_size,
        device=config.device,
    )

    # Start generation and playback threads
    stop_event = threading.Event()

    def generation_loop():
        while not stop_event.is_set():
            tile = generator.generate_tile()
            audio = TilePipeline.tile_to_audio(tile, config)
            audio_engine.add_segment(audio)
            time.sleep(0.1)  # small delay to avoid busy loop

    gen_thread = threading.Thread(target=generation_loop, daemon=True)
    gen_thread.start()

    audio_engine.start()

    # Set up control interfaces
    controllers = []
    if not args.no_keyboard:
        keyboard = KeyboardController(generator, config)
        keyboard.start()
        controllers.append(keyboard)
    if not args.no_midi:
        midi = MidiController(generator, config)
        midi.start()
        controllers.append(midi)

    print("Diffusion Music Box running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        audio_engine.stop()
        for c in controllers:
            c.stop()
        gen_thread.join(timeout=2)
        print("Stopped.")


if __name__ == "__main__":
    main()
