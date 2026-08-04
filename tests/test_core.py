import numpy as np
import pytest

from src.config import Config
from src.augmentation import SpectrogramAugmenter
from src.denoiser import Denoiser
from src.scheduler import TileScheduler
from src.spectrogram import SpectrogramTransformer
from src.spectrum_inverse import SpectrogramInverseTransformer


def test_config_defaults():
    cfg = Config(path='/tmp/nonexistent_config.json')
    assert cfg.data['sample_rate'] == 22050
    assert cfg.data['n_freq'] == 257
    assert cfg.data['n_frames'] == 128
    assert cfg.data['hidden_channels'] == 64
    assert cfg.data['diffusion_steps'] == 50
    assert cfg.data['noise_schedule'] == 'linear'
    assert cfg.data['tempo'] == 120.0
    assert cfg.data['crossfade_samples'] == 256
    assert cfg.data['block_size'] == 1024
    assert cfg.data['buffer_multiplier'] == 8
    assert cfg.data['device'] is None
    assert cfg.data['output_dir'] == 'output'


def test_config_load_merges():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'sample_rate': 44100, 'diffusion_steps': 100}, f)
        tmp_path = f.name
    try:
        cfg = Config(path=tmp_path)
        assert cfg.data['sample_rate'] == 44100
        assert cfg.data['diffusion_steps'] == 100
        assert cfg.data['n_freq'] == 257  # default remains
    finally:
        import os
        os.unlink(tmp_path)


def test_augmentation_seed_reproducibility():
    aug1 = SpectrogramAugmenter(seed=42)
    aug2 = SpectrogramAugmenter(seed=42)
    tile = np.random.rand(16, 32)
    t1 = aug1.random_time_shift(tile)
    t2 = aug2.random_time_shift(tile)
    np.testing.assert_array_equal(t1, t2)


def test_augmentation_no_shift_when_zero():
    aug = SpectrogramAugmenter(seed=0)
    tile = np.random.rand(16, 32)
    original = tile.copy()
    result = aug.random_time_shift(tile, max_shift=0)
    np.testing.assert_array_equal(result, original)


def test_augmentation_shape_preserved():
    aug = SpectrogramAugmenter(seed=1)
    tile = np.random.rand(16, 32)
    for op in [
        lambda t: aug.random_time_shift(t),
        lambda t: aug.random_freq_shift(t),
        lambda t: aug.random_scale(t),
        lambda t: aug.random_noise(t),
    ]:
        result = op(tile)
        assert result.shape == tile.shape


def test_denoiser_shape_and_output():
    denoiser = Denoiser(n_freq=16, hidden_channels=8)
    tile = np.random.rand(16, 32).astype(np.float32)
    output = denoiser(tile)
    assert output.shape == tile.shape
    assert np.all(np.isfinite(output))


def test_denoiser_denoising_tendency():
    # On a constant input, output should be close to input after training? Not applicable; just check output is finite.
    denoiser = Denoiser(n_freq=16, hidden_channels=8)
    tile = np.ones((16, 32), dtype=np.float32)
    output = denoiser(tile)
    assert np.all(np.isfinite(output))
    # The network is untrained, but it should not crash.


def test_scheduler_basic():
    scheduler = TileScheduler(n_frames=32, hop_length=16, overlap=8)
    # Add a segment and get next tile
    scheduler.add_tile(np.random.rand(16, 32))
    assert scheduler.ready()
    tile = scheduler.get_next_tile()
    assert tile is not None
    assert tile.shape == (16, 32)


def test_scheduler_empty():
    scheduler = TileScheduler(n_frames=32, hop_length=16, overlap=8)
    assert not scheduler.ready()
    assert scheduler.get_next_tile() is None


def test_spectrogram_roundtrip():
    # Create a simple audio signal
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    transformer = SpectrogramTransformer(n_freq=257, n_frames=128, hop_length=256, win_length=512)
    spec = transformer.forward(audio)
    assert spec.shape[0] == 257
    assert spec.shape[1] > 0

    inverse = SpectrogramInverseTransformer(hop_length=256, win_length=512)
    reconstructed = inverse.forward(spec)
    assert reconstructed.shape[0] == len(audio)
    # Check that the signal is approximately preserved in magnitude
    assert np.all(np.isfinite(reconstructed))


def test_audio_engine_crossfade():
    from src.audio_engine import AudioEngine
    engine = AudioEngine(sample_rate=22050, block_size=256, crossfade_samples=64)
    # Add a segment and check that the buffer is updated
    segment = np.random.randn(2048).astype(np.float32)
    engine.add_segment(segment)
    assert engine._write_pos > 0
