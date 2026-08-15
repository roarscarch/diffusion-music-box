import json
import os
from typing import Any, Dict


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required fields."""


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON configuration file.

    Parameters
    ----------
    path : str
        Path to the JSON config file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    ConfigError
        If the file does not exist, is not valid JSON, or is not a dictionary.
    """
    if not os.path.exists(path):
        raise ConfigError(f"Config file not found: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a JSON object, got {type(data).__name__}")
    return data


def validate_required_keys(config: Dict[str, Any], required: tuple) -> None:
    """Ensure all required keys are present in the config.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate.
    required : tuple
        Tuple of key names that must be present.

    Raises
    ------
    ConfigError
        If any required key is missing.
    """
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing required config keys: {', '.join(missing)}")


def merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two config dictionaries, with override taking precedence.

    Parameters
    ----------
    default : Dict[str, Any]
        Base configuration.
    override : Dict[str, Any]
        Overrides applied on top.

    Returns
    -------
    Dict[str, Any]
        Merged configuration.
    """
    result = default.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def deep_merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for merge_configs that mutates and returns base.

    This function is kept for compatibility with potential callers that expect
    in-place merging. It modifies ``base`` in place and returns it.

    Parameters
    ----------
    base : Dict[str, Any]
        Base dictionary to merge into.
    extra : Dict[str, Any]
        Dictionary whose values override base.

    Returns
    -------
    Dict[str, Any]
        The merged base dictionary.
    """
    for key, value in extra.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_preset(preset_name: str, presets_dir: str = "presets") -> Dict[str, Any]:
    """Load a named preset from a JSON file.

    Parameters
    ----------
    preset_name : str
        Name of the preset (without .json extension).
    presets_dir : str, optional
        Directory where preset files are stored.

    Returns
    -------
    Dict[str, Any]
        The preset configuration.

    Raises
    ------
    ConfigError
        If the preset file is missing or invalid.
    """
    path = os.path.join(presets_dir, f"{preset_name}.json")
    return load_config(path)
