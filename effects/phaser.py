"""Phaser: cascaded all-pass filters with LFO-modulated break frequency."""
from __future__ import annotations

import numpy as np


class Phaser:
    name = "Phaser"

    def __init__(
        self,
        n_stages: int = 4,
        lfo_freq: float = 0.5,
        center: float = 800.0,
        depth: float = 600.0,
        feedback: float = 0.3,
        mix: float = 0.5,
    ):
        self.n_stages = int(n_stages)
        self.lfo_freq = float(lfo_freq)
        self.center = float(center)
        self.depth = float(depth)
        self.feedback = float(feedback)
        self.mix = float(mix)

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        n = len(audio)
        out = np.zeros(n, dtype=np.float32)
        # All-pass states (one per stage)
        states_x = np.zeros(self.n_stages, dtype=np.float32)
        states_y = np.zeros(self.n_stages, dtype=np.float32)
        fb = 0.0

        for i in range(n):
            lfo = 0.5 * (1.0 + np.sin(2.0 * np.pi * self.lfo_freq * i / fs))
            fc = self.center + self.depth * (lfo - 0.5) * 2.0
            fc = max(20.0, min(fs / 2.0 - 100.0, fc))
            # First-order all-pass coefficient: a = (tan(pi*fc/fs)-1)/(tan(pi*fc/fs)+1)
            tan_v = np.tan(np.pi * fc / fs)
            a = (tan_v - 1.0) / (tan_v + 1.0)

            x = audio[i] + self.feedback * fb
            v = x
            for s in range(self.n_stages):
                y = a * v + states_x[s] - a * states_y[s]
                states_x[s] = v
                states_y[s] = y
                v = y
            fb = v
            out[i] = (1 - self.mix) * audio[i] + self.mix * v

        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out = out / peak
        return out

    def reset_state(self) -> None:
        pass
