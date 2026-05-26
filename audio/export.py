from __future__ import annotations

import numpy as np
from scipy.io import wavfile


def export_wav(audio: np.ndarray, fs: int, path: str) -> None:
    """Normalize and save audio as 16-bit WAV."""
    if audio.size == 0:
        raise ValueError("Audio array is empty")
    peak = float(np.max(np.abs(audio)))
    audio_norm = audio / peak if peak > 0 else audio
    audio_int16 = (audio_norm * 32767.0).astype(np.int16)
    wavfile.write(path, fs, audio_int16)
