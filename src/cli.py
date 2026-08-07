import argparse
import sys

from .config import load_config
from .main import run_generator


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Diffusion Music Box - AI-generated ambient music from noise"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON/YAML config file (optional)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Audio sample rate (default from config or 22050)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=None,
        help="Audio block size in samples (default from config or 1024)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Output audio device (name or index, default system default)",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable interactive keyboard control",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    return parser


def main(argv=None) -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    # Load config if provided
    config = load_config(args.config) if args.config else {}

    # Merge CLI overrides
    if args.sample_rate is not None:
        config["sample_rate"] = args.sample_rate
    if args.blocksize is not None:
        config["block_size"] = args.blocksize
    if args.device is not None:
        config["device"] = args.device

    # Run the generator
    try:
        run_generator(config, use_keyboard=not args.no_keyboard)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
