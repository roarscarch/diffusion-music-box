import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class AudioConfig:
    """Configuration for audio engine parameters."""
    sample_rate: int = 22050
    block_size: int = 1024
    crossfade_samples: int = 256
    device: Optional[str] = None


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model parameters."""
    steps: int = 50
    noise_schedule: str = "linear"
    noise_scale: float = 1.0
    seed: Optional[int] = None


@dataclass
class GeneratorConfig:
    """Configuration for tile generation and overlap-add."""
    tile_width: int = 128
    tile_height: int = 256
    hop_length: int = 64
    overlap: int = 32
    fft_size: int = 1024
    sample_rate: int = 22050


@dataclass
class KeyboardConfig:
    """Configuration for interactive keyboard controller."""
    enabled: bool = False
    midi_channel: int = 0
    octave_offset: int = 0


@dataclass
class MidiConfig:
    """Configuration for MIDI controller."""
    enabled: bool = False
    port_name: Optional[str] = None


@dataclass
class AppConfig:
    """Top-level configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    midi: MidiConfig = field(default_factory=MidiConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create config from a dictionary, ignoring unknown keys nested."""
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            diffusion=DiffusionConfig(**data.get("diffusion", {})),
            generator=GeneratorConfig(**data.get("generator", {})),
            keyboard=KeyboardConfig(**data.get("keyboard", {})),
            midi=MidiConfig(**data.get("midi", {}