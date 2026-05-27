"""Reverb / echo effects (Section 3.1).

Ported from sound_effects_alumnos.ipynb (echo, reverb) with return-type fix
(list -> np.ndarray) and stateful class wrappers. Multi-tap stub is completed.

Each class exposes a class-level ``PARAMS`` list — the GUI's effects panel
introspects it to auto-build knob widgets:

    PARAMS = [(attr_name, min, max, step, label), ...]
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import scipy.signal as sp


class Echo:
    """Single delayed copy added to the input.

    Direct port of ``apply_echo`` from sound_effects_alumnos.ipynb:

        wet[n] = x[n] + alpha * x[n - delay_samples]   for n > delay_samples
        wet[n] = x[n]                                   otherwise

    The final output crossfades the dry input with the wet path:

        y[n] = (1 - mix) * x[n] + mix * wet[n]
    """
    name = "Echo"
    PARAMS = [
        ("delay", 0.0, 2.0, 0.01, "Delay (s)"),
        ("alpha", 0.0, 1.5, 0.05, "Alpha"),
        ("mix",   0.0, 1.0, 0.05, "Dry/Wet"),
    ]

    def __init__(self, delay: float = 0.5, alpha: float = 0.3, mix: float = 1.0):
        self.delay = float(delay)
        self.alpha = float(alpha)
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        delay_samples = int(self.delay * fs)
        n = audio.size
        wet = np.empty(n, dtype=np.float32)
        for i in range(n):
            if i > delay_samples:
                wet[i] = audio[i] + self.alpha * audio[i - delay_samples]
            else:
                wet[i] = audio[i]
        return ((1.0 - self.mix) * audio + self.mix * wet).astype(np.float32)

    def reset_state(self) -> None:
        pass


class Reverb:
    """Single-delay feedback reverb.

    Direct port of ``apply_reverb`` from sound_effects_alumnos.ipynb:

        wet[n] = x[n] + alpha * wet[n - delay_samples]   for n > delay_samples
        wet[n] = x[n]                                     otherwise

    Crossfaded with the dry input via the ``mix`` parameter.
    """
    name = "Reverb"
    PARAMS = [
        ("delay", 0.0, 1.0, 0.01, "Delay (s)"),
        ("alpha", 0.0, 0.95, 0.05, "Alpha"),
        ("mix",   0.0, 1.0, 0.05, "Dry/Wet"),
    ]

    def __init__(self, delay: float = 0.1, alpha: float = 0.7, mix: float = 1.0):
        self.delay = float(delay)
        self.alpha = float(np.clip(alpha, 0.0, 0.95))
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        delay_samples = int(self.delay * fs)
        n = audio.size
        wet = np.empty(n, dtype=np.float32)
        for i in range(n):
            if i > delay_samples:
                wet[i] = audio[i] + self.alpha * wet[i - delay_samples]
            else:
                wet[i] = audio[i]
        return ((1.0 - self.mix) * audio + self.mix * wet).astype(np.float32)

    def reset_state(self) -> None:
        pass


class MultiTapReverb:
    """Plate-like multi-tap delay reverb (completes the notebook stub).

    Five fixed taps with adjustable maximum delay and overall decay scale.
    """
    name = "Multi-Tap Reverb"
    PARAMS = [
        ("max_delay", 0.05, 1.5, 0.05, "Max delay (s)"),
        ("decay",     0.1,  1.5, 0.05, "Decay scale"),
    ]

    def __init__(self, max_delay: float = 0.37, decay: float = 1.0):
        self.max_delay = float(max_delay)
        self.decay = float(decay)

    def _taps(self) -> tuple[list[float], list[float]]:
        # Geometric spacing of taps up to max_delay; decays drop ~halving.
        base_delays = [0.135, 0.30, 0.51, 0.73, 1.0]
        base_decays = [0.7, 0.5, 0.35, 0.22, 0.12]
        delays = [d * self.max_delay for d in base_delays]
        decays = [c * self.decay for c in base_decays]
        return delays, decays

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        delays, decays = self._taps()
        max_d = max(1, int(max(delays) * fs))
        out = np.zeros(len(audio) + max_d, dtype=np.float32)
        out[: len(audio)] = audio
        for dly, gain in zip(delays, decays):
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
    PARAMS = [
        ("wet", 0.0, 1.0, 0.05, "Dry/Wet"),
    ]

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
