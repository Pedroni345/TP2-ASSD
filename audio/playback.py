from __future__ import annotations

import numpy as np
import sounddevice as sd


class AudioPlayer:
    def __init__(self):
        self._playing = False

    def play(self, audio: np.ndarray, fs: int) -> None:
        sd.play(audio, fs)
        self._playing = True

    def stop(self) -> None:
        sd.stop()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing
