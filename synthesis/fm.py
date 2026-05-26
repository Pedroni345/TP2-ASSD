"""FM synthesis (Exercise 2.4, optional).

Classic single-carrier / single-modulator FM with time-varying modulation index,
following Chowning's framework. Default preset is a clarinet (n/m = 3/2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import MidiNote, midi_to_freq, overlap_add_notes


@dataclass(frozen=True)
class FMPreset:
    name: str
    n: int          # carrier ratio
    m: int          # modulator ratio
    I_max: float    # peak modulation index
    attack: float
    decay: float
    sustain: float
    release: float


FM_PRESETS: dict[str, FMPreset] = {
    "clarinet": FMPreset(name="clarinet", n=3, m=2, I_max=2.0, attack=0.06, decay=0.05, sustain=0.95, release=0.10),
    "brass": FMPreset(name="brass", n=1, m=1, I_max=4.0, attack=0.03, decay=0.10, sustain=0.85, release=0.15),
    "bell": FMPreset(name="bell", n=1, m=1, I_max=8.0, attack=0.003, decay=0.5, sustain=0.3, release=1.0),
    "wood": FMPreset(name="wood", n=1, m=3, I_max=3.0, attack=0.005, decay=0.15, sustain=0.5, release=0.20),
}


def _adsr(fs: int, duration: float, A: float, D: float, S: float, R: float) -> np.ndarray:
    n_total = max(1, int(fs * duration))
    t = np.linspace(0, duration, n_total)
    t_off = max(0.0, duration - R)
    xs = [0.0, A, A + D, t_off, duration]
    ys = [0.0, 1.0, S, S, 0.0]
    return np.interp(t, xs, ys).astype(np.float32)


def fm_synthesize(
    f0: float,
    duration: float,
    fs: int,
    preset: FMPreset,
) -> np.ndarray:
    """Generate one FM note. f0 = fundamental (gcd of fc and fm scales)."""
    # f0 = gcd(fc, fm) means fc = n*f0, fm = m*f0
    fc = preset.n * f0
    fm = preset.m * f0

    n_total = max(1, int(fs * duration))
    t = np.arange(n_total) / fs

    A_env = _adsr(fs, duration, preset.attack, preset.decay, preset.sustain, preset.release)
    # Modulation index: ramp up during attack, hold at I_max, decay during release
    I_env = _adsr(fs, duration, preset.attack, preset.decay, preset.sustain, preset.release) * preset.I_max

    # x(t) = A(t) * cos(2*pi*fc*t + I(t)*cos(2*pi*fm*t - pi/2))
    phase_m = 2.0 * np.pi * fm * t - np.pi / 2.0
    phase = 2.0 * np.pi * fc * t + I_env * np.cos(phase_m)
    x = A_env * np.cos(phase)
    return x.astype(np.float32)


class FMSynth:
    """FM synthesizer with selectable preset."""

    def __init__(self, preset: FMPreset | str = "clarinet"):
        if isinstance(preset, str):
            if preset not in FM_PRESETS:
                raise KeyError(f"Unknown FM preset: {preset}")
            preset = FM_PRESETS[preset]
        self.preset = preset
        self.name = f"FM · {preset.name.title()}"

    def synthesize_note(self, note: MidiNote, fs: int = 44100) -> np.ndarray:
        duration = max(note.duration + self.preset.release, 0.05)
        f0 = midi_to_freq(note.pitch)
        audio = fm_synthesize(f0, duration, fs, self.preset)
        amp = note.velocity / 127.0
        return (audio * amp).astype(np.float32)

    def render_track(self, notes: list[MidiNote], fs: int = 44100) -> np.ndarray:
        return overlap_add_notes(self, notes, fs, tail=self.preset.release + 0.2)
