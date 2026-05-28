"""PSOLA synthesis (Exercise 2.1).

Core algorithms ported from the corrected notebook "TP2 ASSD version2.ipynb"
(cells 4 and 6). Two fixes over the first version:
  1. The OLA grain is now ~20 fundamental periods wide (mitad_ventana = periodo*10)
     instead of ~2, which removes the metallic/comb-filter reconstruction artifact.
  2. pitch_shift folds the requested note duration into the PSOLA stretch factor,
     so pitch and length are produced together in one stretch+resample.
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

    # Window half-width spans ~10 fundamental periods on each side. The first
    # version used a single period here, which produced too-short grains and a
    # metallic comb-filter artifact under overlap-add.
    mitad_ventana = periodo * 10

    for nuevo_pm in nuevos_pitch_marks:
        pm_original_ideal = nuevo_pm / factor_estiramiento
        idx_cercano = np.argmin(np.abs(pitch_marks - pm_original_ideal))
        pm_real = int(pitch_marks[idx_cercano])

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
    duracion_deseada_seg: float,
    fs: int,
) -> np.ndarray:
    """Pitch-shift by N semitones and fit to a target duration in one pass.

    The requested duration is folded into the PSOLA stretch factor, then a
    single resample restores the pitch ratio while setting the exact length:

        factor_psola = (longitud_deseada / longitud_original) * factor_frecuencia
        resample(estirada, longitud_deseada)  ->  pitch * factor_frecuencia
    """
    longitud_original = len(senal)
    longitud_deseada = int(duracion_deseada_seg * fs)
    if longitud_deseada == 0 or longitud_original == 0:
        return np.array([], dtype=np.float64)

    factor_frecuencia = 2.0 ** (semitonos / 12.0)
    factor_duracion = longitud_deseada / longitud_original
    factor_psola = factor_duracion * factor_frecuencia

    senal_estirada = psola_time_stretch(senal, pitch_marks, factor_psola, fs)
    return signal.resample(senal_estirada, longitud_deseada)


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
        # Pitch-shift AND fit duration in one pass, at the sample's native fs so
        # the tuning stays correct regardless of the project sample rate.
        audio = pitch_shift(
            self.sample, self.pitch_marks, semitonos, note.duration, self._sample_fs
        )
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        # Convert the sample rate to the target fs if they differ.
        if self._sample_fs != fs:
            new_len = max(1, int(len(audio) * fs / self._sample_fs))
            audio = signal.resample(audio, new_len)
        # Enforce exact length and apply a short fade-out to avoid clicks.
        muestras_deseadas = max(1, int(note.duration * fs))
        if len(audio) >= muestras_deseadas:
            audio = audio[:muestras_deseadas].copy()
            fade = min(int(0.01 * fs), len(audio) // 4)
            if fade > 0:
                audio[-fade:] *= np.linspace(1.0, 0.0, fade)
        else:
            audio = np.concatenate([audio, np.zeros(muestras_deseadas - len(audio))])
        # Apply velocity
        amp = note.velocity / 127.0
        return (audio * amp).astype(np.float32)

    def render_track(self, notes: list[MidiNote], fs: int = 44100) -> np.ndarray:
        return overlap_add_notes(self, notes, fs, tail=0.5)
