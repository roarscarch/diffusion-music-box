import numpy as np


class PhaseVocoder:
    """Phase vocoder for time-stretching and pitch-shifting audio segments.

    This module provides a lightweight phase vocoder implementation that can
    be used to alter the tempo and pitch of generated audio segments in
    real-time. It is designed to work with the spectrogram-based diffusion
    pipeline, allowing the user to adjust the perceived tempo and pitch of
    the ambient output without regenerating the audio.

    The implementation uses the standard phase vocoder algorithm with a
    windowed STFT, phase propagation, and overlap-add synthesis. It supports
    both time-stretching (rate > 1 slows down, rate < 1 speeds up) and
    pitch-shifting (by resampling after time-stretch).

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio.
    fft_size : int
        FFT size for analysis/synthesis.
    hop_size : int
        Hop size for analysis.
    """

    def __init__(self, sample_rate=22050, fft_size=1024, hop_size=256):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.analysis_window = np.hanning(fft_size).astype(np.float32)
        self.synthesis_window = np.hanning(fft_size).astype(np.float32)
        # Normalize synthesis window to avoid amplitude modulation
        self.synthesis_window = self.synthesis_window / np.sqrt(np.sum(self.synthesis_window**2))

    def _stft(self, audio):
        """Compute the short-time Fourier transform of the audio.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.

        Returns
        -------
        tuple
            (spectrogram, phases) where spectrogram is complex array of shape
            (num_frames, num_bins) and phases is the unwrapped phase array.
        """
        num_frames = 1 + (len(audio) - self.fft_size) // self.hop_size
        if num_frames <= 0:
            num_frames = 1
        spectrogram = np.zeros((num_frames, self.fft_size // 2 + 1), dtype=np.complex64)
        phases = np.zeros((num_frames, self.fft_size // 2 + 1), dtype=np.float32)
        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.fft_size]
            if len(frame) < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - len(frame)))
            windowed = frame * self.analysis_window
            spectrum = np.fft.rfft(windowed)
            spectrogram[i] = spectrum
            phases[i] = np.angle(spectrum)
        return spectrogram, phases

    def _istft(self, spectrogram, num_samples=None):
        """Inverse STFT to reconstruct audio from a complex spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            Complex array of shape (num_frames, num_bins).
        num_samples : int, optional
            Desired output length. If None, computed from frames.

        Returns
        -------
        np.ndarray
            1D float array of reconstructed audio.
        """
        num_frames, num_bins = spectrogram.shape
        if num_samples is None:
            num_samples = (num_frames - 1) * self.hop_size + self.fft_size
        output = np.zeros(num_samples, dtype=np.float32)
        window_sum = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_frames):
            start = i * self.hop_size
            if start >= num_samples:
                break
            spectrum = spectrogram[i]
            frame = np.fft.irfft(spectrum, n=self.fft_size)
            frame = frame * self.synthesis_window
            end = min(start + self.fft_size, num_samples)
            length = end - start
            output[start:end] += frame[:length]
            window_sum[start:end] += self.synthesis_window[:length]
        # Avoid division by zero
        mask = window_sum > 1e-8
        output[mask] /= window_sum[mask]
        return output

    def time_stretch(self, audio, rate):
        """Time-stretch the audio by the given rate.

        Parameters
        ----------
        audio : np.ndarray
            1D float array of audio samples.
        rate : float
            Time-stretch factor. >1 slows down, <1 speeds up.

        Returns
        -------
        np.ndarray
            Time-stretched audio.
        """
        if rate <= 0:
            raise ValueError("Rate must be positive")
        if len(audio) < self.fft_size:
            return audio.copy()

        spectrogram, _ = self._stft(audio)
        num_frames, num_bins = spectrogram.shape
        # Analysis hop size is fixed; synthesis hop is analysis_hop * rate
        synth_hop = self.hop_size * rate
        # Number of output frames for the same duration
        out_frames = int(num_frames / rate)
        if out_frames < 1:
            out_frames = 1

        # Initialize phase accumulator
        phase_acc = np.zeros(num_bins, dtype=np.float32)
        # Angular frequency increments per bin (radians per hop)
        freqs = np.fft.rfftfreq(self.fft_size, d=1.0/self.sample_rate)
        omega = 2.0 * np.pi * freqs * self.hop_size / self.sample_rate

        out_spec = np.zeros((out_frames, num_bins), dtype=np.complex64)
        # Process each output frame using the nearest input frame
        for i in range(out_frames):
            # Source frame index (linear mapping)
            src_idx = int(i * rate)
            src_idx = min(src_idx, num_frames - 1)

            if src_idx == 0:
                # First frame: copy magnitude, keep original phase
                mag = np.abs(spectrogram[0])
                phase = np.angle(spectrogram[0])
            else:
                # Compute phase difference between consecutive input frames
                phase_prev = np.angle(spectrogram[src_idx - 1])
                phase_curr = np.angle(spectrogram[src_idx])
                delta_phi = phase_curr - phase_prev
                # Wrap to [-pi, pi]
                delta_phi = np.mod(delta_phi + np.pi, 2*np.pi) - np.pi
                # Expected phase advance from the bin frequency
                expected = omega
                # Phase deviation from expected
                deviation = delta_phi - expected
                # Wrap deviation
                deviation = np.mod(deviation + np.pi, 2*np.pi) - np.pi
                # True frequency deviation (in radians per hop)
                true_deviation = deviation / self.hop_size
                # New phase increment for synthesis hop
                phase_increment = omega + true_deviation * synth_hop
                phase_acc += phase_increment
                mag = np.abs(spectrogram[src_idx])
                phase = phase_acc

            out_spec[i] = mag * np.exp(1j * phase)

        # Synthesize audio
        output = self._istft(out_spec)
        # Trim to match expected duration
        expected_len = int(len(audio) * rate)
        if len(output) > expected_len:
            output = output[:expected_len]
        elif len(output) < expected_len:
            output = np.pad(output, (0, expected_len - len(output)))
        return output

    def pitch_shift(self, audio, semitones):
        """Pitch-shift the audio by a number of semitones.

        This is implemented by time-stretching the audio without changing
        pitch, then resampling to restore the original duration, effectively
        shifting the pitch.

        Parameters
        ----------
        audio : np.ndarray
            1D float array