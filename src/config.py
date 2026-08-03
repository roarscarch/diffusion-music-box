import json
import os
from pathlib import Path


class Config:
    """Load and manage configuration settings from a JSON file.

    The configuration file is expected to be at ~/.diffusion_music_box/config.json
    by default, but a custom path can be provided. Any missing keys are filled
    with defaults. The configuration is reloadable at runtime.
    """

    DEFAULTS = {
        "sample_rate": 22050,
        "n_freq": 257,
        "n_frames": 128,
        "hidden_channels": 64,
        "diffusion_steps": 50,
        "noise_schedule": "linear",
        "tempo": 120.0,
        "crossfade_samples": 256,
        "block_size": 1024,
        "buffer_multiplier": 8,
        "device": None,
        "output_dir": "output",
    }

    def __init__(self, path=None):
        self.path = Path(path) if path else Path.home() / ".diffusion_music_box" / "config.json"
        self.data = dict(self.DEFAULTS)
        self.load()

    def load(self):
        """Load configuration from file, merging with defaults."""
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
                # Merge user data over defaults
                self.data.update({k: v for k, v in user_data.items() if k in self.DEFAULTS})
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not load config from {self.path}: {e}")
        else:
            self.save()

    def save(self):
        """Write current configuration to disk."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except OSError as e:
            print(f"Warning: could not save config to {self.path}: {e}")

    def get(self, key):
        """Get a configuration value."""
        return self.data[key]

    def set(self, key, value):
        """Set a configuration value and persist it."""
        if key in self.DEFAULTS:
            self.data[key] = value
            self.save()
        else:
            raise KeyError(f"Unknown config key: {key}")

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)
