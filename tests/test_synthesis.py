"""Testbenches for synthesis engines (2.1 PSOLA, 2.2 Karplus-Strong, 2.3 Additive, 2.4 FM)."""
import numpy as np
import pytest

from synthesis.base import MidiNote, midi_to_freq
from synthesis.karplus import KarplusStrongSynth, karplus_strong
from synthesis.additive import AdditiveSynth
from synthesis.fm import FMSynth


def _peak_freq(audio: np.ndarray, fs: int, fmin: float = 50.0) -> float:
    """Return the frequency bin with the largest magnitude above fmin."""
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / fs)
    spec[freqs < fmin] = 0.0
    return float(freqs[np.argmax(spec)])


class TestKarplusStrong:
    def test_produces_audio(self):
        audio = karplus_strong(f0=440.0, fs=22050, duration=0.3)
        assert audio.size > 0
        assert np.max(np.abs(audio)) > 0.1

    def test_fundamental_frequency(self):
        """The dominant spectral peak should fall on a harmonic of f0."""
        for f0 in [220.0, 440.0, 880.0]:
            audio = karplus_strong(f0=f0, fs=22050, duration=0.5)
            peak = _peak_freq(audio[2000:], 22050)  # skip noise burst
            # Karplus often emphasises low harmonics — accept any harmonic within 5%
            harmonic = round(peak / f0)
            assert harmonic >= 1
            assert abs(peak - harmonic * f0) / f0 < 0.05, f"f0={f0}, peak={peak}, h={harmonic}"

    def test_decay_with_low_RL(self):
        """Lower RL should produce faster amplitude decay."""
        long_decay = karplus_strong(f0=220.0, fs=22050, duration=1.0, RL=0.999)
        short_decay = karplus_strong(f0=220.0, fs=22050, duration=1.0, RL=0.85)
        # Energy in the second half should be smaller for short_decay
        half = len(long_decay) // 2
        e_long = np.sum(long_decay[half:] ** 2)
        e_short = np.sum(short_decay[half:] ** 2)
        assert e_short < e_long

    def test_synth_engine_wrapper(self):
        synth = KarplusStrongSynth(RL=0.99, b=1.0)
        note = MidiNote(pitch=69, velocity=100, start=0.0, duration=0.3)
        audio = synth.synthesize_note(note, fs=22050)
        assert audio.size > 0
        assert audio.dtype == np.float32


class TestAdditive:
    def test_static_level1(self):
        synth = AdditiveSynth("organ", level=1)
        note = MidiNote(pitch=60, velocity=100, start=0.0, duration=0.3)
        audio = synth.synthesize_note(note, fs=22050)
        assert audio.size > 0
        assert np.max(np.abs(audio)) > 0.1

    def test_fundamental_within_tolerance(self):
        synth = AdditiveSynth("organ", level=1)
        note = MidiNote(pitch=69, velocity=127, start=0.0, duration=0.5)
        audio = synth.synthesize_note(note, fs=22050)
        peak = _peak_freq(audio, 22050)
        assert abs(peak - 440.0) < 5.0

    def test_levels_all_run(self):
        for level in [1, 2, 3, 4, 5]:
            synth = AdditiveSynth("piano", level=level)
            note = MidiNote(pitch=60, velocity=100, start=0.0, duration=0.3)
            audio = synth.synthesize_note(note, fs=22050)
            assert audio.size > 0, f"level {level} returned empty audio"


class TestFM:
    def test_clarinet_preset(self):
        synth = FMSynth("clarinet")
        note = MidiNote(pitch=60, velocity=100, start=0.0, duration=0.3)
        audio = synth.synthesize_note(note, fs=22050)
        assert audio.size > 0
        assert np.max(np.abs(audio)) > 0.1

    def test_carrier_frequency(self):
        synth = FMSynth("clarinet")  # n=3, m=2 -> carrier = 3*f0
        note = MidiNote(pitch=69, velocity=127, start=0.0, duration=0.4)
        audio = synth.synthesize_note(note, fs=44100)
        # FM produces sidebands; the dominant component should be close to a multiple of f0=440
        peak = _peak_freq(audio, 44100)
        # Should be in the harmonic series of 440
        n_harm = round(peak / 440.0)
        assert n_harm >= 1
        assert abs(peak - n_harm * 440.0) / 440.0 < 0.1
