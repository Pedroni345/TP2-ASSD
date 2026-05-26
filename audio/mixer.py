from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pretty_midi

from synthesis.base import MidiNote, SynthEngine
from effects.base import EffectChain


@dataclass
class TrackConfig:
    synth: Optional[SynthEngine] = None
    effects: EffectChain = field(default_factory=EffectChain)
    gain: float = 1.0
    mute: bool = False
    solo: bool = False


class Mixer:
    def __init__(self):
        self.tracks: dict[int, TrackConfig] = {}
        self.master_effects: EffectChain = EffectChain()

    def set_track(self, channel: int, config: TrackConfig) -> None:
        self.tracks[channel] = config

    def render(self, midi: pretty_midi.PrettyMIDI, fs: int = 44100) -> np.ndarray:
        if not midi.instruments:
            return np.zeros(1, dtype=np.float32)

        any_solo = any(
            cfg.solo for cfg in self.tracks.values() if cfg.synth is not None
        )

        total_end = midi.get_end_time() + 1.0
        total_samples = int(total_end * fs) + 1
        master = np.zeros(total_samples, dtype=np.float32)

        for idx, instrument in enumerate(midi.instruments):
            cfg = self.tracks.get(idx)
            if cfg is None or cfg.synth is None or cfg.mute:
                continue
            if any_solo and not cfg.solo:
                continue

            notes = [
                MidiNote(
                    pitch=n.pitch,
                    velocity=n.velocity,
                    start=n.start,
                    duration=n.end - n.start,
                )
                for n in instrument.notes
            ]
            if not notes:
                continue

            try:
                track_audio = cfg.synth.render_track(notes, fs)
            except Exception as e:
                print(f"[Mixer] Track {idx} synth error: {e}")
                continue

            if cfg.effects.effects:
                track_audio = cfg.effects.process(track_audio, fs)

            track_audio = track_audio * cfg.gain
            n_samples = min(track_audio.size, master.size)
            master[:n_samples] += track_audio[:n_samples]

        if self.master_effects.effects:
            master = self.master_effects.process(master, fs)

        peak = float(np.max(np.abs(master)))
        if peak > 1.0:
            master = master / peak
        return master
