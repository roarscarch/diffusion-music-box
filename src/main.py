#!/usr/bin/env python3
"""Main entry point for the Diffusion Music Box.

Runs the full pipeline: generate spectrogram tiles with the diffusion denoiser,
convert them to audio via the spectrogram inverter, and play them back with the
audio engine. Keyboard controls adjust parameters in real-time.
"""

import argparse
import sys
import time
import numpy as np

from src.audio_engine import AudioEngine
from src.denoiser import Denoiser
from src.midi_controller import MIDIKeyboardController
from src.spectrogram import SpectrogramInverter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate endless ambient music from noise using a tiny diffusion model."
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Audio sample rate (default: 22050)",
    )
    parser.add_argument(
        "--n-freq",
        type=int,
        default=257,
        help="Number of frequency bins in the spectrogram (default: 257)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=128,
        help="Number of time frames per spectrogram tile (default: 128)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=50,
        help="Number of diffusion steps per tile (default: 50)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.6,
        help="Noise scale for diffusion (default: 0.6)",
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=120.0,
        help="Tempo in BPM for tile generation rate (default: 120.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Output device name or index (default: None)",
    )
    return parser.parse_args()


def generate_tile(denoiser, steps, noise_scale, rng):
    """Generate a single spectrogram tile via denoising.

    Starts from pure noise and iteratively denoises it.
    """
    shape = (denoiser.n_freq, denoiser.n_frames)
    # Start with pure noise
    x = rng.normal(0, 1.0, shape).astype(np.float32)
    for i in range(steps):
        # In a real diffusion model, we'd schedule noise removal.
        # Here we simply apply the denoiser to the current tile.
        # This is a placeholder: the denoiser's forward method returns
        # the denoised image.
        # We'll implement a simple reverse diffusion approximation.
        # For now, just add a bit of noise each step and denoise.
        noise = rng.normal(0, noise_scale, shape).astype(np.float32)
        x = x + noise
        x = denoiser.denoise(x)  # assuming denoiser has a denoise method
    return x


def main():
    args = parse_args()

    # Initialize components
    denoiser = Denoiser(n_freq=args.n_freq, n_frames=args.n_frames)
    inverter = SpectrogramInverter(
        sample_rate=args.sample_rate, n_freq=args.n_freq, n_frames=args.n_frames
    )
    audio_engine = AudioEngine(
        sample_rate=args.sample_rate, device=args.device
    )
    controller = MIDIKeyboardController(
        initial_params={
            "diffusion_steps": args.diffusion_steps,
            "noise_scale": args.noise_scale,
            "tempo": args.tempo,
        }
    )

    rng = np.random.default_rng(0)

    print("Diffusion Music Box started.")
    print("Controls: n/N steps, s/S noise, t/T tempo, space pause, q quit")

    # Start playback
    audio_engine.start()
    controller.start()

    try:
        while not controller.get_params()["quit"]:
            params = controller.get_params()
            if not params["paused"]:
                # Generate a new tile
                tile = generate_tile(
                    denoiser, params["diffusion_steps"], params["noise_scale"], rng
                )
                # Convert to audio
                audio = inverter.synthesize(tile)
                # Queue for playback
                audio_engine.add_segment(audio)

                # Compute sleep time based on tempo (beats per minute)
                # Each tile corresponds to roughly one beat?
                # For simplicity, we generate at a fixed rate.
                bpm = params["tempo"]
                sleep_time = 60.0 / bpm  # seconds per beat
                time.sleep(sleep_time)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        controller.stop()
        audio_engine.stop()


if __name__ == "__main__":
    main()
