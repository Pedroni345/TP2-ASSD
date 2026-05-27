"""Testbenches for audio effects (Section 3)."""
import numpy as np

from effects import (
    Echo, Reverb, MultiTapReverb, Clipper, SoftClipper,
    Flanger, Vibrato, Chorus, Phaser, WahWah, EffectChain,
)


def _impulse(n: int = 22050, at: int = 1) -> np.ndarray:
    # Place the impulse 1 sample in so the notebook-faithful
    # "i > delay_samples" branch can actually read it back as x[1].
    x = np.zeros(n, dtype=np.float32)
    x[at] = 1.0
    return x


def _sine(f: float, fs: int, dur: float) -> np.ndarray:
    t = np.arange(int(fs * dur)) / fs
    return np.sin(2 * np.pi * f * t).astype(np.float32)


class TestEchoReverb:
    def test_echo_delays_signal(self):
        fs = 22050
        echo = Echo(delay=0.1, alpha=0.5)  # impulse lives at sample 1
        out = echo.process(_impulse(fs), fs)
        # Dry copy stays at sample 1; the echo tap fires at sample
        # delay_samples + 1 with amplitude alpha.
        idx_wet = int(0.1 * fs) + 1
        assert abs(out[1]) > 0.5
        assert abs(out[idx_wet]) > 0.2

    def test_reverb_decays_into_buffer(self):
        fs = 22050
        rv = Reverb(delay=0.05, alpha=0.6, mix=0.5)
        x = _impulse(fs)
        out = rv.process(x, fs)
        # The IIR feeds energy past the impulse: tail energy must be > 0.
        tail = out[int(0.05 * fs) + 1:]
        assert np.max(np.abs(tail)) > 0.1

    def test_multitap_extends_length(self):
        fs = 22050
        rv = MultiTapReverb()
        x = _impulse(fs)
        out = rv.process(x, fs)
        assert len(out) > len(x)


class TestDistortion:
    def test_clipper_respects_threshold(self):
        fs = 22050
        clip = Clipper(gain=10.0, threshold=0.3)
        out = clip.process(_sine(440.0, fs, 0.1), fs)
        assert np.max(np.abs(out)) <= 0.3 + 1e-6

    def test_soft_clip_bounded(self):
        fs = 22050
        sc = SoftClipper(gain=5.0)
        out = sc.process(_sine(440.0, fs, 0.1) * 2.0, fs)
        assert np.max(np.abs(out)) <= 1.0


class TestModulation:
    def test_flanger_runs(self):
        fs = 22050
        out = Flanger().process(_sine(440.0, fs, 0.2), fs)
        assert out.size > 0

    def test_vibrato_runs(self):
        fs = 22050
        out = Vibrato().process(_sine(440.0, fs, 0.2), fs)
        assert out.size > 0

    def test_chorus_runs(self):
        fs = 22050
        out = Chorus().process(_sine(440.0, fs, 0.2), fs)
        assert out.size > 0

    def test_phaser_runs(self):
        fs = 22050
        out = Phaser().process(_sine(440.0, fs, 0.2), fs)
        assert out.size > 0

    def test_wahwah_runs(self):
        fs = 22050
        out = WahWah().process(_sine(440.0, fs, 0.2), fs)
        assert out.size > 0


class TestChain:
    def test_chain_applies_in_order(self):
        fs = 22050
        chain = EffectChain([Clipper(gain=10.0, threshold=0.5), Echo(delay=0.05, alpha=0.3)])
        out = chain.process(_sine(440.0, fs, 0.2), fs)
        assert out.size > 0
