import argparse
import sys
from .config import load_config, save_config
from .presets import list_presets, load_preset, save_preset
from .noise_schedule import NoiseSchedule
from .audio_engine import AudioEngine
from .generator import Generator
from .keyboard_controller import KeyboardController
from .midi_controller import MidiController
from .tempo_scheduler import TempoScheduler
from .spectrogram import Spectrogram
from .spectrum_inverse import SpectrumInverse
from .tile_pipeline import TilePipeline
from .overlap_add import OverlapAdd


def parse_args():
    """Parse command-line arguments for the diffusion music box.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Generate endless ambient music using diffusion on spectrograms.'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Sample rate for audio output (default: 22050)'
    )
    parser.add_argument(
        '--fft-size',
        type=int,
        default=1024,
        help='FFT size for spectrogram (default: 1024)'
    )
    parser.add_argument(
        '--hop-length',
        type=int,
        default=256,
        help='Hop length for spectrogram (default: 256)'
    )
    parser.add_argument(
        '--diffusion-steps',
        type=int,
        default=20,
        help='Number of diffusion steps per tile (default: 20)'
    )
    parser.add_argument(
        '--noise-level',
        type=float,
        default=0.5,
        help='Initial noise level for diffusion (default: 0.5)'
    )
    parser.add_argument(
        '--tempo',
        type=int,
        default=60,
        help='Tempo in BPM (default: 60)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Output audio device name or index (default: system default)'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=1024,
        help='Audio block size in samples (default: 1024)'
    )
    parser.add_argument(
        '--crossfade-samples',
        type=int,
        default=256,
        help='Crossfade samples between segments (default: 256)'
    )
    parser.add_argument(
        '--preset',
        type=str,
        default=None,
        help='Load a preset by name'
    )
    parser.add_argument(
        '--save-preset',
        type=str,
        default=None,
        metavar='NAME',
        help='Save current settings as a preset and exit'
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available presets and exit'
    )
    parser.add_argument(
        '--no-keyboard',
        action='store_true',
        help='Disable keyboard control'
    )
    parser.add_argument(
        '--no-midi',
        action='store_true',
        help='Disable MIDI control'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable spectrum visualizer (requires matplotlib)'
    )
    return parser.parse_args()


def main():
    """Entry point for the CLI."""
    args = parse_args()

    # Handle preset listing
    if args.list_presets:
        presets = list_presets()
        if not presets:
            print("No presets found.")
        else:
            print("Available presets:")
            for p in presets:
                print(f"  {p}")
        return

    # Load configuration
    config = load_config(args.config)
    # Override config with CLI args
    config['sample_rate'] = args.sample_rate
    config['fft_size'] = args.fft_size
    config['hop_length'] = args.hop_length
    config['diffusion_steps'] = args.diffusion_steps
    config['noise_level'] = args.noise_level
    config['tempo'] = args.tempo
    config['device'] = args.device
    config['block_size'] = args.block_size
    config['crossfade_samples'] = args.crossfade_samples

    # Load preset if specified
    if args.preset:
        try:
            preset = load_preset(args.preset)
            config.update(preset)
        except FileNotFoundError:
            print(f"Error: Preset '{args.preset}' not found.", file=sys.stderr)
            sys.exit(1)

    # Save preset if requested
    if args.save_preset:
        save_preset(args.save_preset, config)
        print(f"Preset '{args.save_preset}' saved.")
        return

    # Initialize components
    sample_rate = config['sample_rate']
    fft_size = config['fft_size']
    hop_length = config['hop_length']
    diffusion_steps = config['diffusion_steps']
    noise_level = config['noise_level']
    tempo = config['tempo']

    noise_schedule = NoiseSchedule(
        steps=diffusion_steps,
        initial_noise=noise_level
    )

    spectrogram = Spectrogram(
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_length=hop_length
    )

    spectrum_inverse = SpectrumInverse(
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_length=hop_length
    )

    tile_pipeline = TilePipeline(
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_length=hop_length
    )

    overlap_add = OverlapAdd(
        hop_length=hop_length,
        window_length=fft_size
    )

    generator = Generator(
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_length=hop_length,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule
    )

    audio_engine = AudioEngine(
        sample_rate=sample_rate,
        block_size=config['block_size'],
        crossfade_samples=config['crossfade_samples'],
        device=config['device']
    )

    # Initialize controllers
    controllers = []
    if not args.no_keyboard:
        keyboard = KeyboardController()
        controllers.append(keyboard)
    if not args.no_midi:
        try:
            midi = MidiController()
            controllers.append(midi)
        except Exception as e:
            print(f"Warning: MIDI controller unavailable: {e}", file=sys.stderr)

    # Tempo scheduler
    tempo_scheduler = TempoScheduler(
        bpm=tempo,
        sample_rate=sample_rate,
        hop_length=hop_length
    )

    print("Starting Diffusion Music Box...")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"FFT size: {fft_size}")
    print(f"Hop length: {hop_length}")
    print(f"Diffusion steps: {diffusion_steps}")
    print(f"Noise level: {noise_level}")
    print(f"Tempo: {tempo} BPM")
    if controllers:
        print("Controls:")
        for c in controllers:
            print(f"  {c.description}