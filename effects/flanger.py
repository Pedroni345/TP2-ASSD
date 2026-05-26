"""Flanger family: Flanger, Vibrato, Chorus.

Ported and parameterized from sound_effects_alumnos.ipynb cell 17.
The hardcoded 500 Hz LFO is now configurable.
"""
from __future__ import annotations

import numpy as np


def _modulated_delay(
    audio: np.ndarray,
    fs: int,
    base_delay: float,
    depth: float,
    lfo_freq: float,
    feedback: float,
    mix: float,
) -> np.ndarray:
    """Common delay-line core used by Flanger/Vibrato/Chorus."""
    n = len(audio)
    max_delay_samples = int((base_delay + depth) * fs) + 2
    buf = np.zeros(n + max_delay_samples, dtype=np.float32)
    buf[:n] = audio
    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        lfo = 0.5 * (1.0 + np.cos(2.0 * np.pi * lfo_freq * i / fs))
        d_samples = (base_delay + depth * lfo) * fs
        d_int = int(d_samples)
        d_frac = d_samples - d_int
        src_idx = i - d_int
        if src_idx > 0:
            a = buf[src_idx]
            b = buf[src_idx - 1] if src_idx - 1 >= 0 else 0.0
            delayed = (1 - d_frac) * a + d_frac * b
        else:
            delayed = 0.0
        y = audio[i] + mix * delayed
        out[i] = y
        # apply feedback into the buffer
        if feedback > 0 and i + d_int < len(buf):
            buf[i + d_int] += feedback * out[i]

    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out = out / peak
    return out


class Flanger:
    name = "Flanger"

    def __init__(self, depth: float = 0.005, lfo_freq: float = 0.4, feedback: float = 0.5, mix: float = 0.7):
        self.depth = float(depth)
        self.lfo_freq = float(lfo_freq)
        self.feedback = float(feedback)
        self.mix = float(mix)

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        return _modulated_delay(audio, fs, base_delay=0.001, depth=self.depth,
                                lfo_freq=self.lfo_freq, feedback=self.feedback, mix=self.mix)

    def reset_state(self) -> None:
        pass


class Vibrato:
    name = "Vibrato"

    def __init__(self, depth: float = 0.003, lfo_freq: float = 5.0):
        self.depth = float(depth)
        self.lfo_freq = float(lfo_freq)

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        # Pure pitch modulation: only the wet signal, no dry, no feedback
        n = len(audio)
        max_delay_samples = int(self.depth * fs * 2) + 2
        out = np.zeros(n, dtype=np.float32)
        for i in range(n):
            lfo = 0.5 * (1.0 + np.sin(2.0 * np.pi * self.lfo_freq * i / fs))
            d_samples = self.depth * fs * lfo
            d_int = int(d_samples)
            d_frac = d_samples - d_int
            idx = i - d_int
            if idx > 0:
                a = audio[idx]
                b = audio[idx - 1] if idx - 1 >= 0 else 0.0
                out[i] = (1 - d_frac) * a + d_frac * b
        return out

    def reset_state(self) -> None:
        pass


class Chorus:
    """Multi-voice chorus: 3 detuned voices summed with the dry signal."""
    name = "Chorus"

    def __init__(self, depth: float = 0.005, lfo_freq: float = 0.8, mix: float = 0.5):
        self.depth = float(depth)
        self.lfo_freq = float(lfo_freq)
        self.mix = float(mix)

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        v1 = _modulated_delay(audio, fs, base_delay=0.020, depth=self.depth,
                              lfo_freq=self.lfo_freq, feedback=0.0, mix=1.0)
        v2 = _modulated_delay(audio, fs, base_delay=0.025, depth=self.depth * 0.8,
                              lfo_freq=self.lfo_freq * 1.13, feedback=0.0, mix=1.0)
        v3 = _modulated_delay(audio, fs, base_delay=0.018, depth=self.depth * 1.1,
                              lfo_freq=self.lfo_freq * 0.87, feedback=0.0, mix=1.0)
        wet = (v1 + v2 + v3) / 3.0
        out = (1 - self.mix) * audio + self.mix * wet
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32)

    def reset_state(self) -> None:
        pass
