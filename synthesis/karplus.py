"""Karplus-Strong synthesis (Exercise 2.2).

Core algorithm ported from TP2 ASSD.ipynb cell 31.
Includes both string (b=1) and percussion (b<1) modes, fractional-delay tuning.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import scipy.signal as sp
import soundfile as sf

from .base import MidiNote, midi_to_freq, overlap_add_notes


def karplus_strong(
    f0: float,
    fs: int,
    RL: float = 0.99,
    b: float = 1.0,
    duration: float = 1.0,
    noisedur: float = 0.05,
    noisetype: str = "normal",
    box_resonance: bool = False,
    ir_path: str | None = None,
) -> np.ndarray:
    """Single-note Karplus-Strong synthesis with fractional-delay tuning."""
    L_exact = fs / f0 - 0.5
    M = max(2, int(np.floor(L_exact)))
    eta = L_exact - M

    N = int(duration * fs)
    n_noise = int(noisedur * fs)
    if n_noise >= N:
        n_noise = N - 1
    if noisetype == "blanco":
        noise = np.random.rand(n_noise)
    elif noisetype == "normal":
        noise = np.random.randn(n_noise)
    else:
        raise ValueError("noisetype must be 'blanco' or 'normal'")

    x = np.concatenate((noise, np.zeros(N - n_noise)))
    xA = np.zeros(N)
    y = np.zeros(N)

    for n in range(N):
        xA[n] = x[n] + RL * y[n - M] if n >= M else x[n]
        sign = 1 if np.random.rand() < b else -1
        y[n] = sign * (0.5 * xA[n] + 0.5 * xA[n - 1] if n > 0 else 0.5 * xA[n])

    # Fractional-delay linear interpolation
    y = sp.lfilter([1 - eta, eta], [1.0], y)

    if box_resonance and ir_path:
        try:
            ir, ir_fs = sf.read(ir_path)
            if ir.ndim > 1:
                ir = ir[:, 0]
            if ir_fs != fs:
                g = gcd(fs, ir_fs)
                ir = sp.resample_poly(ir, fs // g, ir_fs // g)
            y = sp.convolve(y, ir)[:N]
        except Exception as e:
            print(f"[Karplus] IR convolution skipped: {e}")

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


class KarplusStrongSynth:
    """Plucked-string / percussion synth via Karplus-Strong."""

    name: str = "Karplus-Strong"

    def __init__(
        self,
        RL: float = 0.99,
        b: float = 1.0,
        noisetype: str = "normal",
        box_resonance: bool = False,
        ir_path: str | None = None,
        tail: float = 0.0,
    ):
        self.RL = RL
        self.b = b
        self.noisetype = noisetype
        self.box_resonance = box_resonance
        self.ir_path = ir_path
        # extra audio time appended after MIDI note duration so the string can ring out
        self.tail = tail

    def synthesize_note(self, note: MidiNote, fs: int = 44100) -> np.ndarray:
        f0 = midi_to_freq(note.pitch)
        total_duration = note.duration + self.tail
        if total_duration <= 0:
            return np.zeros(1, dtype=np.float32)
        audio = karplus_strong(
            f0=f0,
            fs=fs,
            RL=self.RL,
            b=self.b,
            duration=total_duration,
            noisetype=self.noisetype,
            box_resonance=self.box_resonance,
            ir_path=self.ir_path,
        )
        # Apply a release fade so notes can decay naturally instead of cutting off
        n_held = int(note.duration * fs)
        if self.tail > 0 and len(audio) > n_held:
            fade_len = len(audio) - n_held
            fade = np.linspace(1.0, 0.0, fade_len) ** 2
            audio[n_held:] = audio[n_held:] * fade
        amp = note.velocity / 127.0
        return (audio * amp).astype(np.float32)

    def render_track(self, notes: list[MidiNote], fs: int = 44100) -> np.ndarray:
        return overlap_add_notes(self, notes, fs, tail=max(0.5, self.tail))
