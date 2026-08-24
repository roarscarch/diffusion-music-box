import numpy as np


class PhaseVocoder:
    """Phase vocoder for time-stretching and pitch-shifting audio.

    This module implements a classic phase vocoder algorithm that can
    independently control the time scale and pitch of an audio signal.
    It is useful in the diffusion music box for adjusting the tempo of
    generated segments or for creating pitch-shifted variations.

    The implementation uses an STFT-based approach: the input signal is
    transformed into a time-frequency representation, the phases are
    corrected to preserve horizontal coherence, and the frames are
    resampled in time. Pitch shifting is achieved by resampling the
    frequency axis.

    Parameters
    ----------
    fft_size : int
        FFT size in samples.
    hop_size : int
        Hop size in samples between analysis frames.
    """

    def __init__(self, fft_size=1024, hop_size=256):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window = np.hanning(fft_size).astype(np.float32)

    def stretch(self, audio, time_scale=1.0):
        """Time-stretch the audio by a given factor.

        Parameters
        ----------
        audio : np.ndarray
            Input audio as a 1D float array.
        time_scale : float, optional
            Time-stretch factor. >1 slows down, <1 speeds up.
            Must be positive.

        Returns
        -------
        np.ndarray
            Time-stretched audio as a 1D float array.
        """
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be 1D")

        # Analysis STFT
        n_frames = max(1, (len(audio) - self.fft_size) // self.hop_size + 1)
        frames = np.zeros((n_frames, self.fft_size), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_size
            frames[i] = audio[start:start + self.fft_size] * self.window
        spec = np.fft.rfft(frames, axis=1)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # Phase propagation for horizontal coherence
        n_bins = spec.shape[1]
        phase_advance = 2 * np.pi * np.arange(n_bins) * self.hop_size / self.fft_size
        phase_acc = phase[0].copy()
        phase_corrected = np.zeros_like(phase)
        phase_corrected[0] = phase_acc
        for i in range(1, n_frames):
            delta = phase[i] - phase[i-1] - phase_advance
            delta = np.mod(delta + np.pi, 2 * np.pi) - np.pi
            phase_acc += phase_advance + delta
            phase_corrected[i] = phase_acc

        # Resample frames in time
        n_out_frames = int(round(n_frames / time_scale))
        out_spec = np.zeros((n_out_frames, n_bins), dtype=np.complex64)
        for i in range(n_out_frames):
            src_idx = i * time_scale
            idx0 = int(np.floor(src_idx))
            idx1 = min(idx0 + 1, n_frames - 1)
            frac = src_idx - idx0
            interp_mag = mag[idx0] * (1 - frac) + mag[idx1] * frac
            interp_phase = phase_corrected[idx0] * (1 - frac) + phase_corrected[idx1] * frac
            out_spec[i] = interp_mag * np.exp(1j * interp_phase)

        # Inverse STFT with overlap-add
        out_frames = np.fft.irfft(out_spec, n=self.fft_size, axis=1).astype(np.float32)
        out_len = (n_out_frames - 1) * self.hop_size + self.fft_size
        out_audio = np.zeros(out_len, dtype=np.float32)
        for i in range(n_out_frames):
            start = i * self.hop_size
            out_audio[start:start + self.fft_size] += out_frames[i] * self.window
        return out_audio

    def shift_pitch(self, audio, semitones=0.0):
        """Pitch-shift the audio by a given number of semitones.

        This is achieved by resampling the frequency axis of the STFT
        representation. A positive value raises the pitch, negative lowers.

        Parameters
        ----------
        audio : np.ndarray
            Input audio as a 1D float array.
        semitones : float, optional
            Pitch shift in semitones (can be fractional).

        Returns
        -------
        np.ndarray
            Pitch-shifted audio as a 1D float array.
        """
        if semitones == 0:
            return np.asarray(audio, dtype=np.float32).copy()
        ratio = 2.0 ** (semitones / 12.0)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be 1D")

        # STFT analysis
        n_frames = max(1, (len(audio) - self.fft_size) // self.hop_size + 1)
        frames = np.zeros((n_frames, self.fft_size), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_size
            frames[i] = audio[start:start + self.fft_size] * self.window
        spec = np.fft.rfft(frames, axis=1)
        n_bins = spec.shape[1]

        # Resample frequency bins
        bin_freqs = np.arange(n_bins) * (self.fft_size / self.fft_size)  # normalized 0..0.5
        # For each output bin, find source bin by dividing by ratio
        out_bins = np.arange(n_bins)
        src_bins = out_bins / ratio
        src_bins = np.clip(src_bins, 0, n_bins - 1)
        idx0 = np.floor(src_bins).astype(int)
        idx1 = np.minimum(idx0 + 1, n_bins - 1)
        frac = src_bins - idx0
        # Interpolate magnitude and phase
        mag = np.abs(spec)
        phase = np.angle(spec)
        new_mag = mag[:, idx0] * (1 - frac) + mag[:, idx1] * frac
        new_phase = phase[:, idx0] * (1 - frac) + phase[:, idx1] * frac
        new_spec = new_mag * np.exp(1j * new_phase)

        # Inverse STFT
        out_frames = np.fft.irfft(new_spec, n=self.fft_size, axis=1).astype(np.float32)
        out_len = (n_frames - 1) * self.hop_size + self.fft_size
        out_audio = np.zeros(out_len, dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_size
            out_audio[start:start + self.fft_size] += out_frames[i] * self.window
        return out_audio
