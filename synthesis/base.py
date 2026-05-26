from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MidiNote:
    pitch: int       # MIDI note 0-127
    velocity: int    # 0-127
    start: float     # seconds
    duration: float  # seconds


def midi_to_freq(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


class SynthEngine(Protocol):
    name: str

    def synthesize_note(self, note: MidiNote, fs: int = 44100) -> np.ndarray: ...

    def render_track(self, notes: list[MidiNote], fs: int = 44100) -> np.ndarray: ...


def overlap_add_notes(
    synth: SynthEngine,
    notes: list[MidiNote],
    fs: int = 44100,
    tail: float = 0.5,
) -> np.ndarray:
    """Render each note via synth.synthesize_note and overlap-add onto a timeline."""
    if not notes:
        return np.zeros(1, dtype=np.float32)
    total_end = max(n.start + n.duration for n in notes) + tail
    total_samples = int(total_end * fs) + 1
    out = np.zeros(total_samples, dtype=np.float32)
    for note in notes:
        audio = synth.synthesize_note(note, fs)
        if audio.size == 0:
            continue
        start_idx = int(note.start * fs)
        end_idx = min(start_idx + audio.size, total_samples)
        out[start_idx:end_idx] += audio[: end_idx - start_idx].astype(np.float32)
    return out
