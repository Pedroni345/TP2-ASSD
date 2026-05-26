"""Reverb / echo effects (Section 3.1).

Ported from sound_effects_alumnos.ipynb (echo, reverb) with return-type fix
(list -> np.ndarray) and stateful class wrappers. Multi-tap stub is completed.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import scipy.signal as sp


class Echo:
    name = "Echo"

    def __init__(self, delay: float = 0.3, feedback: float = 0.4):
        self.delay = float(delay)
        self.feedback = float(feedback)

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        d = max(1, int(self.delay * fs))
        out = np.zeros(len(audio) + d, dtype=np.float32)
        out[: len(audio)] = audio
        out[d : d + len(audio)] += self.feedback * audio
        return out

    def reset_state(self) -> None:
        pass


class Reverb:
    """Single-delay feedback reverb (IIR style from notebook cell 13)."""
    name = "Reverb"

    def __init__(self, delay: float = 0.08, feedback: float = 0.6, mix: float = 0.5):
        self.delay = float(delay)
        self.feedback = float(np.clip(feedback, 0.0, 0.95))
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        d = max(1, int(self.delay * fs))
        out = np.zeros(len(audio) + d * 8, dtype=np.float32)
        out[: len(audio)] = audio
        for n in range(d, len(out)):
            out[n] = out[n] + self.feedback * out[n - d]
        wet = out[: len(audio) + d]
        peak = float(np.max(np.abs(wet)))
        if peak > 0:
            wet = wet / peak
        dry = np.zeros_like(wet)
        dry[: len(audio)] = audio
        return ((1 - self.mix) * dry + self.mix * wet).astype(np.float32)

    def reset_state(self) -> None:
        pass


class MultiTapReverb:
    """Plate-like multi-tap delay reverb (completes the notebook stub)."""
    name = "Multi-Tap Reverb"

    def __init__(
        self,
        delays: list[float] | None = None,
        decays: list[float] | None = None,
    ):
        self.delays = delays if delays is not None else [0.05, 0.11, 0.19, 0.27, 0.37]
        self.decays = decays if decays is not None else [0.7, 0.5, 0.35, 0.22, 0.12]

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        max_d = max(1, int(max(self.delays) * fs))
        out = np.zeros(len(audio) + max_d, dtype=np.float32)
        out[: len(audio)] = audio
        for dly, gain in zip(self.delays, self.decays):
            d = max(1, int(dly * fs))
            out[d : d + len(audio)] += gain * audio
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out = out / peak
        return out

    def reset_state(self) -> None:
        pass


class ConvolutionReverb:
    """Convolutional reverb: convolve with impulse response file."""
    name = "Convolution Reverb"

    def __init__(self, ir_path: str, wet: float = 0.5):
        self.ir_path = ir_path
        self.wet = float(np.clip(wet, 0.0, 1.0))
        self._ir: np.ndarray | None = None
        self._ir_fs: int | None = None

    def _load(self, fs: int) -> None:
        ir, ir_fs = sf.read(self.ir_path)
        if ir.ndim > 1:
            ir = np.mean(ir, axis=1)
        if ir_fs != fs:
            from math import gcd
            g = gcd(int(fs), int(ir_fs))
            ir = sp.resample_poly(ir, fs // g, ir_fs // g)
        self._ir = ir.astype(np.float32)
        self._ir_fs = fs

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        if self._ir is None or self._ir_fs != fs:
            self._load(fs)
        wet = sp.fftconvolve(audio, self._ir, mode="full")
        peak = float(np.max(np.abs(wet)))
        if peak > 0:
            wet = wet / peak
        out_len = len(wet)
        dry = np.zeros(out_len, dtype=np.float32)
        dry[: len(audio)] = audio
        return ((1 - self.wet) * dry + self.wet * wet).astype(np.float32)

    def reset_state(self) -> None:
        pass
