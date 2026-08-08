import json
import os
import tempfile
from pathlib import Path


class PresetManager:
    """Manage named presets for parameter configurations.

    Presets are stored as JSON files in a user-specific directory
    (e.g., ~/.diffusion-music-box/presets/). Each preset contains
    a dictionary of parameter values that can be applied to the
    generator or scheduler.
    """

    def __init__(self, preset_dir=None):
        """Initialize the preset manager.

        Parameters
        ----------
        preset_dir : str or Path, optional
            Directory where presets are stored. If None, use the default
            user config directory.
        """
        if preset_dir is None:
            preset_dir = Path.home() / '.diffusion-music-box' / 'presets'
        self.preset_dir = Path(preset_dir)
        self.preset_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name, params):
        """Save a preset with the given name and parameters.

        Parameters
        ----------
        name : str
            Name of the preset (used as filename, must be a valid filename).
        params : dict
            Dictionary of parameter values to save.

        Raises
        ------
        ValueError
            If the name is empty or contains invalid characters.
        """
        if not name or not name.replace('_', '').isalnum():
            raise ValueError("Preset name must contain only letters, numbers, and underscores")
        filepath = self.preset_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2, sort_keys=True)

    def load(self, name):
        """Load a preset by name.

        Parameters
        ----------
        name : str
            Name of the preset.

        Returns
        -------
        dict
            The parameter dictionary.

        Raises
        ------
        FileNotFoundError
            If the preset does not exist.
        """
        filepath = self.preset_dir / f"{name}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Preset '{name}' not found")
        with open(filepath, 'r') as f:
            return json.load(f)

    def list(self):
        """List all available presets.

        Returns
        -------
        list of str
            Sorted list of preset names (without .json extension).
        """
        return sorted([p.stem for p in self.preset_dir.glob('*.json')])

    def delete(self, name):
        """Delete a preset by name.

        Parameters
        ----------
        name : str
            Name of the preset to delete.

        Raises
        ------
        FileNotFoundError
            If the preset does not exist.
        """
        filepath = self.preset_dir / f"{name}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Preset '{name}' not found")
        filepath.unlink()

    def get_default_dir(self):
        """Return the default preset directory as a string."""
        return str(self.preset_dir)
