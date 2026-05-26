"""PSOLA synthesis (Exercise 2.1).

Core algorithms ported from TP2 ASSD.ipynb cells 4 and 6.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.signal as signal
import soundfile as sf

from .base import MidiNote, midi_to_freq, overlap_add_notes


def estimar_f0_autocorrelacion(senal: np.ndarray, fs: int) -> tuple[float, int]:
    """Estimates fundamental frequency and period via autocorrelation."""
    corr = signal.correlate(senal, senal, mode="full")
    corr = corr[len(corr) // 2 :]
    min_lag = int(fs / 1000.0)
    max_lag = min(int(fs / 60.0), len(corr))
    pico_lag = np.argmax(corr[min_lag:max_lag]) + min_lag
    f0 = fs / pico_lag
    return f0, int(pico_lag)


def detectar_pitch_marks(senal: np.ndarray, periodo_muestras: int) -> np.ndarray:
    """Finds peaks spaced ~1 period apart."""
    picos, _ = signal.find_peaks(senal, distance=periodo_muestras * 0.8)
    return picos


def psola_time_stretch(
    senal: np.ndarray,
    pitch_marks: np.ndarray,
    factor_estiramiento: float,
    fs: int,
) -> np.ndarray:
    """PSOLA time-stretching. factor>1 lengthens, factor<1 shortens."""
    if len(pitch_marks) < 2:
        return senal.copy()

    nueva_longitud = int(len(senal) * factor_estiramiento)
    senal_salida = np.zeros(nueva_longitud, dtype=np.float64)
    periodo = int(pitch_marks[1] - pitch_marks[0])
    nuevos_pitch_marks = np.arange(0, nueva_longitud, periodo)

    for nuevo_pm in nuevos_pitch_marks:
        pm_original_ideal = nuevo_pm / factor_estiramiento
        idx_cercano = np.argmin(np.abs(pitch_marks - pm_original_ideal))
        pm_real = int(pitch_marks[idx_cercano])
        mitad_ventana = periodo

        inicio_orig = max(0, pm_real - mitad_ventana)
        fin_orig = min(len(senal), pm_real + mitad_ventana)
        inicio_nuevo = max(0, int(nuevo_pm) - mitad_ventana)
        fin_nuevo = inicio_nuevo + (fin_orig - inicio_orig)

        if fin_nuevo > nueva_longitud:
            fin_orig -= fin_nuevo - nueva_longitud
            fin_nuevo = nueva_longitud
        if fin_orig <= inicio_orig:
            continue

        ventana = np.hanning(fin_orig - inicio_orig)
        segmento = senal[inicio_orig:fin_orig] * ventana
        senal_salida[inicio_nuevo:fin_nuevo] += segmento

    return senal_salida


def pitch_shift(
    senal: np.ndarray,
    pitch_marks: np.ndarray,
    semitonos: float,
    fs: int,
) -> np.ndarray:
    """Pitch-shift by N semitones via PSOLA time-stretch + resampling."""
    factor_frecuencia = 2.0 ** (semitonos / 12.0)
    senal_estirada = psola_time_stretch(senal, pitch_marks, factor_frecuencia, fs)
    nueva_cantidad = max(1, int(len(senal_estirada) / factor_frecuencia))
    return signal.resample(senal_estirada, nueva_cantidad)


class PSOLASynth:
    """Sample-based synth using PSOLA pitch-shifting."""

    name: str = "PSOLA"

    def __init__(
        self,
        sample_path: str,
        base_midi: int = 60,
        fs: int = 44100,
        trim_start: float = 0.075,
        trim_end: float = 3.0,
    ):
        """Load and analyze the source sample.

        sample_path: WAV/AIF file
        base_midi: MIDI number corresponding to the recorded pitch (C4 = 60)
        """
        audio_raw, sr = sf.read(sample_path)
        if audio_raw.ndim > 1:
            audio_raw = np.mean(audio_raw, axis=1)
        self._sample_fs = sr
        i0 = int(sr * trim_start)
        i1 = min(len(audio_raw), int(sr * trim_end))
        self.sample = audio_raw[i0:i1].astype(np.float64)
        self.base_midi = base_midi
        _, periodo = estimar_f0_autocorrelacion(self.sample, sr)
        self.pitch_marks = detectar_pitch_marks(self.sample, periodo)

    def synthesize_note(self, note: MidiNote, fs: int = 44100) -> np.ndarray:
        semitonos = note.pitch - self.base_midi
        # Pitch-shift at the sample's native fs
        audio = pitch_shift(self.sample, self.pitch_marks, semitonos, self._sample_fs)
        # Resample to target fs if needed
        if self._sample_fs != fs:
            new_len = int(len(audio) * fs / self._sample_fs)
            audio = signal.resample(audio, new_len)
        # Fit duration: truncate or zero-pad
        muestras_deseadas = max(1, int(note.duration * fs))
        if len(audio) > muestras_deseadas:
            audio = audio[:muestras_deseadas]
            # short fade-out to avoid clicks
            fade = min(int(0.01 * fs), len(audio) // 4)
            if fade > 0:
                audio[-fade:] = audio[-fade:] * np.linspace(1.0, 0.0, fade)
        else:
            padding = np.zeros(muestras_deseadas - len(audio))
            audio = np.concatenate([audio, padding])
        # Apply velocity
        amp = note.velocity / 127.0
        return (audio * amp).astype(np.float32)

    def render_track(self, notes: list[MidiNote], fs: int = 44100) -> np.ndarray:
        return overlap_add_notes(self, notes, fs, tail=0.5)
