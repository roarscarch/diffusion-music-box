import numpy as np
import threading
import time
import logging
from src.denoiser import DiffusionDenoiser
from src.spectrogram import SpectrogramProcessor
from src.audio_engine import AudioEngine

logger = logging.getLogger(__name__)


class SegmentScheduler:
    """Schedules generation of spectrogram tiles and feeds them to the audio engine.

    This class coordinates the generation pipeline: it periodically triggers
    the denoiser to produce new spectrogram tiles, converts them to audio
    segments via the spectrogram processor, and passes them to the audio
    engine for seamless playback. It runs in a background thread and can be
    stopped gracefully.

    Parameters
    ----------
    denoiser : DiffusionDenoiser
        Trained denoiser model used for generating spectrogram tiles.
    processor : SpectrogramProcessor
        Processor that converts between spectrograms and audio.
    audio_engine : AudioEngine
        Audio engine that handles playback.
    segment_duration : float, optional
        Duration of each generated audio segment in seconds.
    overlap : float, optional
        Fraction of overlap between consecutive segments (0.0 to 1.0).
    """

    def __init__(self, denoiser, processor, audio_engine, segment_duration=5.0, overlap=0.5):
        self.denoiser = denoiser
        self.processor = processor
        self.audio_engine = audio_engine
        self.segment_duration = segment_duration
        self.overlap = overlap

        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._current_params = {
            "diffusion_steps": 50,
            "noise_schedule": "linear",
            "tempo": 120.0,
        }