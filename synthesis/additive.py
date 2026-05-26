"""Additive synthesis (Exercise 2.3, optional).

Levels:
  1 - Static additive (sum of sines, no envelope)
  2 - Global linear ADSR envelope
  3 - Parametric ADSR with sustain fall-rate
  4 - Per-partial linear envelopes
  5 - Per-partial exponential envelopes

The full analysis pipeline lives in the legacy `additive_synthesis.py`; this module
provides a clean GUI-friendly engine with preset instruments. Custom partials and
envelopes can be supplied at construction time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import MidiNote, midi_to_freq, overlap_add_notes


@dataclass(frozen=True)
class ADSR:
    attack: float = 0.02   # seconds
    decay: float = 0.10
    sustain: float = 0.7   # 0..1
    release: float = 0.15
    note_off_level: float = 0.7  # level at note-off before release
    fall_rate: float = -0.15     # amplitude/second during sustain


@dataclass
class InstrumentPreset:
    name: str
    harmonics: list[float]    # multiplier of f0 (e.g. [1, 2, 3, 4, 5])
    amplitudes: list[float]   # relative amplitude per harmonic
    adsr: ADSR = field(default_factory=ADSR)
    detune_cents: float = 0.0  # inharmonicity in cents

    def freqs_for(self, f0: float) -> np.ndarray:
        return f0 * np.asarray(self.harmonics, dtype=np.float64)


# A few presets pulled from the additive_synthesis.py notebook context.
PRESETS: dict[str, InstrumentPreset] = {
    "organ": InstrumentPreset(
        name="organ",
        harmonics=[1, 2, 3, 4, 5, 6, 8],
        amplitudes=[1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12],
        adsr=ADSR(attack=0.03, decay=0.05, sustain=0.85, release=0.10, note_off_level=0.85, fall_rate=0.0),
    ),
    "piano": InstrumentPreset(
        name="piano",
        harmonics=[1, 2, 3, 4, 5, 6, 7, 8],
        amplitudes=[1.0, 0.55, 0.35, 0.22, 0.18, 0.13, 0.09, 0.06],
        adsr=ADSR(attack=0.005, decay=0.20, sustain=0.40, release=0.30, note_off_level=0.30, fall_rate=-0.5),
    ),
    "bell": InstrumentPreset(
        name="bell",
        harmonics=[1.0, 2.76, 5.40, 8.93, 13.34],
        amplitudes=[1.0, 0.6, 0.4, 0.25, 0.15],
        adsr=ADSR(attack=0.003, decay=0.50, sustain=0.50, release=1.50, note_off_level=0.50, fall_rate=-0.6),
        detune_cents=2.0,
    ),
    "flute": InstrumentPreset(
        name="flute",
        harmonics=[1, 2, 3, 4, 5],
        amplitudes=[1.0, 0.30, 0.15, 0.08, 0.04],
        adsr=ADSR(attack=0.06, decay=0.05, sustain=0.95, release=0.10, note_off_level=0.95, fall_rate=0.0),
    ),
}


def _linear_adsr_envelope(fs: int, duration: float, adsr: ADSR) -> np.ndarray:
    """Level 2/3 linear envelope with optional sustain fall-rate."""
    n_total = max(1, int(fs * duration))

    t_off = max(0.0, duration - adsr.release)
    t_end_attack = min(adsr.attack, t_off)
    t_end_decay = min(t_end_attack + adsr.decay, t_off)

    amp_attack_peak = 1.0
    amp_decay_end = max(adsr.sustain, 1e-3)
    # sustain ramp toward note_off_level applying fall_rate
    sustain_dur = max(0.0, t_off - t_end_decay)
    amp_sustain_end = max(0.0, amp_decay_end + adsr.fall_rate * sustain_dur)
    if amp_sustain_end > amp_decay_end:
        amp_sustain_end = amp_decay_end

    xs = [0.0, t_end_attack, t_end_decay, t_off, duration]
    ys = [0.0, amp_attack_peak, amp_decay_end, amp_sustain_end, 0.0]
    t = np.linspace(0, duration, n_total)
    return np.interp(t, xs, ys).astype(np.float32)


def _exponential_envelope(fs: int, duration: float, adsr: ADSR) -> np.ndarray:
    """Level 5 exponential envelope: exp-rise attack, exp-decay decay/release."""
    n_total = max(1, int(fs * duration))
    t = np.linspace(0, duration, n_total)
    env = np.zeros(n_total, dtype=np.float32)
    t_off = max(0.0, duration - adsr.release)

    for i, ti in enumerate(t):
        if ti < adsr.attack:
            # exponential rise to 1.0
            env[i] = 1.0 - np.exp(-5.0 * ti / max(adsr.attack, 1e-4))
        elif ti < adsr.attack + adsr.decay:
            # exponential decay toward sustain
            x = (ti - adsr.attack) / max(adsr.decay, 1e-4)
            env[i] = adsr.sustain + (1.0 - adsr.sustain) * np.exp(-3.0 * x)
        elif ti < t_off:
            x = ti - (adsr.attack + adsr.decay)
            env[i] = max(0.0, adsr.sustain + adsr.fall_rate * x)
        else:
            x = (ti - t_off) / max(adsr.release, 1e-4)
            env[i] = adsr.sustain * np.exp(-4.0 * x)
    return env


class AdditiveSynth:
    """Additive synthesizer with selectable quality level (1-5)."""

    def __init__(self, preset: InstrumentPreset | str = "organ", level: int = 3):
        if isinstance(preset, str):
            if preset not in PRESETS:
                raise KeyError(f"Unknown preset: {preset}")
            preset = PRESETS[preset]
        self.preset = preset
        self.level = int(np.clip(level, 1, 5))
        self.name = f"Additive · {preset.name.title()} · L{self.level}"

    def synthesize_note(self, note: MidiNote, fs: int = 44100) -> np.ndarray:
        duration = max(note.duration + self.preset.adsr.release, 0.05)
        n_total = max(1, int(fs * duration))
        t = np.arange(n_total) / fs

        f0 = midi_to_freq(note.pitch)
        freqs = self.preset.freqs_for(f0)
        amps = np.asarray(self.preset.amplitudes, dtype=np.float64)
        if self.preset.detune_cents != 0.0:
            detune = (np.random.rand(len(freqs)) - 0.5) * 2.0 * self.preset.detune_cents
            freqs = freqs * (2.0 ** (detune / 1200.0))

        # Level 1: pure sum of sines
        audio = np.zeros(n_total, dtype=np.float64)
        for fk, ak in zip(freqs, amps):
            audio += ak * np.sin(2.0 * np.pi * fk * t)

        # Normalize before envelope
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak

        if self.level >= 2:
            env = _linear_adsr_envelope(fs, duration, self.preset.adsr)
            n = min(len(audio), len(env))
            audio = audio[:n] * env[:n]

        if self.level >= 4:
            # Per-partial envelopes: high harmonics decay faster
            # (already approximated by the global env; we add a per-partial decay multiplier)
            audio = np.zeros(n_total, dtype=np.float64)
            global_env = _linear_adsr_envelope(fs, duration, self.preset.adsr)
            for idx, (fk, ak) in enumerate(zip(freqs, amps)):
                # higher partials decay faster: tau drops with harmonic index
                tau = max(0.05, duration * (0.5 / (1 + 0.3 * idx)))
                partial_decay = np.exp(-t / tau)
                partial = ak * np.sin(2.0 * np.pi * fk * t) * partial_decay
                audio += partial
            audio = audio[: len(global_env)] * global_env[: len(audio)]
            peak = float(np.max(np.abs(audio)))
            if peak > 0:
                audio = audio / peak

        if self.level >= 5:
            env_exp = _exponential_envelope(fs, duration, self.preset.adsr)
            n = min(len(audio), len(env_exp))
            audio = audio[:n] * env_exp[:n]

        amp_vel = note.velocity / 127.0
        return (audio * amp_vel).astype(np.float32)

    def render_track(self, notes: list[MidiNote], fs: int = 44100) -> np.ndarray:
        return overlap_add_notes(self, notes, fs, tail=self.preset.adsr.release + 0.2)
