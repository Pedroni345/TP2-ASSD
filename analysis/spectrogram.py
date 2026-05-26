from __future__ import annotations

import numpy as np
import scipy.signal as signal


def compute_spectrogram(
    audio: np.ndarray,
    fs: int,
    window: str = "hann",
    nperseg: int = 1024,
    overlap_pct: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a spectrogram in dB. Returns (frequencies, times, Sxx_db)."""
    noverlap = int(nperseg * overlap_pct)
    f, t, Sxx = signal.spectrogram(
        audio, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap
    )
    Sxx_db = 10.0 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db
