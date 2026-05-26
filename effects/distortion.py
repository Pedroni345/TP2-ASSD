"""Distortion / clipping effects."""
from __future__ import annotations

import numpy as np


class Clipper:
    """Hard clipping. Ported from sound_effects_alumnos.ipynb cell 15."""
    name = "Clipping"

    def __init__(self, gain: float = 2.0, threshold: float = 0.3):
        self.gain = float(gain)
        self.threshold = float(threshold)

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        y = audio * self.gain
        return np.clip(y, -self.threshold, self.threshold).astype(np.float32)

    def reset_state(self) -> None:
        pass


class SoftClipper:
    """Soft clipping via tanh."""
    name = "Soft Clip"

    def __init__(self, gain: float = 3.0):
        self.gain = float(gain)

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        return np.tanh(self.gain * audio).astype(np.float32)

    def reset_state(self) -> None:
        pass
