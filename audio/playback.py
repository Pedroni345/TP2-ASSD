from __future__ import annotations

import time
import numpy as np
import sounddevice as sd


class AudioPlayer:
    def __init__(self):
        self._playing = False
        self._start_time: float | None = None
        self._start_offset: float = 0.0   # segundos del clip original donde empezó este play()

    def play(self, audio: np.ndarray, fs: int, start_offset: float = 0.0) -> None:
        self._start_offset = start_offset
        self._start_time = time.time()
        sd.play(audio, fs)
        self._playing = True

    def stop(self) -> None:
        sd.stop()
        self._playing = False
        self._start_time = None

    def get_position(self) -> float:
        """Posición de reproducción actual en segundos (relativa al clip original)."""
        if self._start_time is None:
            return self._start_offset
        return self._start_offset + (time.time() - self._start_time)

    @property
    def is_playing(self) -> bool:
        return self._playing
