from __future__ import annotations

from typing import Protocol

import numpy as np


class AudioEffect(Protocol):
    name: str

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray: ...

    def reset_state(self) -> None: ...


class EffectChain:
    def __init__(self, effects: list[AudioEffect] | None = None):
        self.effects: list[AudioEffect] = list(effects) if effects else []

    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray:
        out = audio
        for eff in self.effects:
            out = eff.process(out, fs)
        return out

    def reset_state(self) -> None:
        for eff in self.effects:
            eff.reset_state()

    def add(self, effect: AudioEffect) -> None:
        self.effects.append(effect)

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.effects):
            self.effects.pop(index)

    def clear(self) -> None:
        self.effects.clear()
