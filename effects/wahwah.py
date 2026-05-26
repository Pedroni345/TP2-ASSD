"""Auto-Wah: band-pass filter with LFO-modulated center frequency."""
from __future__ import annotations

import numpy as np
import scipy.signal as sp


class WahWah:
    name = "Wah-Wah"

    def __init__(
        self,
        lfo_freq: float = 1.5,
        center: float = 800.0,
        depth: float = 1200.0,
        Q: float = 4.0,
        mix: float = 0.8,
    ):
        self.lfo_freq = float(lfo_freq)
        self.center = float(center)
        self.depth = float(depth)
        self.Q = float(Q)
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        n = len(audio)
        block = max(64, fs // 100)  # 10 ms blocks: redesign filter ~100 times/sec
        out = np.zeros(n, dtype=np.float32)

        for start in range(0, n, block):
            end = min(start + block, n)
            mid = (start + end) / 2.0
            lfo = 0.5 * (1.0 + np.sin(2.0 * np.pi * self.lfo_freq * mid / fs))
            fc = self.center + self.depth * (lfo - 0.5) * 2.0
            fc = max(50.0, min(fs / 2.0 - 100.0, fc))
            bw = fc / self.Q
            low = max(20.0, fc - bw / 2.0)
            high = min(fs / 2.0 - 1.0, fc + bw / 2.0)
            if high <= low:
                continue
            b, a = sp.butter(2, [low, high], btype="band", fs=fs)
            wet = sp.lfilter(b, a, audio[start:end])
            out[start:end] = ((1 - self.mix) * audio[start:end] + self.mix * wet).astype(np.float32)

        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out = out / peak
        return out

    def reset_state(self) -> None:
        pass
