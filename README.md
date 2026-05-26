# TP2 — MIDI Synthesizer & Spectral Analysis

Desktop application for **25.20 — Análisis de Señales y Sistemas Digitales** (ITBA, 1er cuatrimestre 2026).

The app loads a MIDI file, synthesizes each track with a user-selectable DSP engine (PSOLA / Karplus-Strong / Additive / FM), applies per-track and master-bus audio effects, visualizes the result as a waveform + configurable spectrogram + FL Studio-style timeline, and exports the rendered audio to WAV.

![Architecture](#)
*(Add screenshots here once GitHub renders the repo.)*

---

## How the project fulfills the assignment

The assignment (`TP2 - 25.20 ... Consigna.pdf`) is organised in three sections. This is where each requirement is implemented.

### Section 1 — Spectral Analysis

| § | Requirement | Where it lives |
|---|---|---|
| 1.1 | Cooley-Tukey FFT in C, signature `void fft(float complex *in, float complex *out, size_t N)` | `References/fft.c` (pre-compiled binary `References/Prueba_fft.exe`) |
| 1.1.b | Validation against an O(n²) DFT reference + testbenches | Same file — four `test_*` functions (impulse, sinusoid, in-place, N=1) |
| 1.2.a | Spectrogram with configurable window / segment length / overlap | `analysis/spectrogram.py` + central panel of the GUI |
| 1.2.b–c | Trade-offs and Hann-overlap analysis | Discussed in the report; the GUI lets you change parameters live to demonstrate |

### Section 2 — Instrument Synthesis (mandatory: 2.1, 2.2 · optional: 2.3, 2.4)

| § | Requirement | Implementation |
|---|---|---|
| 2.1 | Sample-based synthesis using **PSOLA** | `synthesis/psola.py` (`PSOLASynth`). Implements autocorrelation-based f0 estimation, pitch-mark detection, PSOLA time-stretching with Hann windows, and pitch-shifting via PSOLA + resampling. Per-note pitch shift is computed from `note.pitch - base_midi`. |
| 2.2 | Physical-model synthesis via **Karplus-Strong** (original and modified) | `synthesis/karplus.py` (`KarplusStrongSynth`). Includes the modified model with the `b` parameter (string ↔ percussion), variable noise type (uniform vs. Gaussian), reflection-loss `RL`, **fractional-delay tuning** (PDF §2.2.f.ii), and optional resonance-box convolution. |
| 2.3 | Additive synthesis (optional) | `synthesis/additive.py` (`AdditiveSynth`). The PDF's five accumulating improvements are exposed as `level` 1–5: <br>• **1**: static `Σ A_k sin(2πf_k t)` <br>• **2**: linear ADSR envelope <br>• **3**: ADSR with sustain fall-rate <br>• **4**: per-partial envelopes <br>• **5**: exponential per-partial envelopes <br>Four instrument presets (organ / piano / bell / flute) provide harmonics, amplitudes and ADSR parameters; `detune_cents` provides inharmonicity (PDF §2.3.e). |
| 2.4 | FM synthesis, clarinet preset (optional) | `synthesis/fm.py` (`FMSynth`). Single carrier + single modulator following Chowning's formulation: `x(t) = A(t) · cos(2π·fc·t + I(t)·cos(2π·fm·t − π/2))`. Presets: clarinet (n=3, m=2), brass, bell, wood. |

All four engines implement a common protocol (`SynthEngine` in `synthesis/base.py`) so they are interchangeable from the mixer's point of view.

### Section 3 — Audio Effects (optional)

| § | Requirement | Implementation |
|---|---|---|
| 3.1 | At least two reverb effects | `effects/reverb.py`: `Echo`, `Reverb` (feedback IIR), `MultiTapReverb` (plate-style), `ConvolutionReverb` (IR-based) |
| 3.2 | At least two flanger-family effects | `effects/flanger.py`: `Flanger`, `Vibrato`, `Chorus` · `effects/phaser.py`: `Phaser` (cascaded all-pass) · `effects/wahwah.py`: `WahWah` (LFO-modulated band-pass) |
| 3.3 | Other effects | `effects/distortion.py`: `Clipper` (hard clip), `SoftClipper` (tanh) |

Effects implement a common `AudioEffect` protocol (`effects/base.py`) and compose into an `EffectChain`. The GUI lets the user build a chain **per track** and an additional **master bus** chain, as required by the PDF.

### Application requirements

| Requirement (from PDF "Objetivos") | Where |
|---|---|
| Open and decode MIDI files | `TP2_Espectro-MIDI.py` (uses `pretty_midi`, explicitly allowed by the PDF) |
| Assign any instrument to any channel | "Tracks" panel — `Método` + `Preset` + `Canal` columns |
| Per-track effects + total-mix effects | Per-track "Edit…" button + "Edit Master Effects…" button |
| Generate a spectrogram from the synthesized audio | Central panel with live window/N/overlap controls |
| Save the audio in a standard format | "Exportar WAV…" button (16-bit PCM via `scipy.io.wavfile`) |
| Test with the *Concierto de Aranjuez* (2nd movement) | `References/Concierto-De-Aranjuez.mid` is bundled; load it via the file dialog and render |

---

## Project layout

```
TP2-ASSD/
├── TP2_Espectro-MIDI.py        # GUI entry point
├── synthesis/                   # Pluggable synthesis engines
│   ├── base.py                  #   SynthEngine protocol, MidiNote dataclass
│   ├── psola.py                 #   2.1 — PSOLA
│   ├── karplus.py               #   2.2 — Karplus-Strong (original + modified)
│   ├── additive.py              #   2.3 — Additive (levels 1-5, 4 presets)
│   └── fm.py                    #   2.4 — FM (clarinet + 3 extras)
├── effects/                     # Pluggable audio effects
│   ├── base.py                  #   AudioEffect protocol, EffectChain
│   ├── reverb.py                #   Echo, Reverb, MultiTap, Convolution
│   ├── distortion.py            #   Clipper, SoftClipper
│   ├── flanger.py               #   Flanger, Vibrato, Chorus
│   ├── phaser.py                #   Cascaded all-pass phaser
│   └── wahwah.py                #   Auto-wah (LFO band-pass)
├── audio/                       # Routing + I/O
│   ├── mixer.py                 #   Multi-track mixer with per-track + master effects
│   ├── playback.py              #   sounddevice wrapper
│   └── export.py                #   WAV export
├── analysis/
│   └── spectrogram.py           # 1.2 — spectrogram computation
├── gui/                         # PyQt6 panels
│   ├── tracks_panel.py          #   Track config table (channel/method/preset/level/gain/effects)
│   ├── timeline_panel.py        #   FL Studio-style event timeline
│   ├── effects_panel.py         #   Effect chain editor dialog
│   └── widgets.py               #   Shared styled widgets
├── tests/                       # pytest testbenches
│   ├── test_synthesis.py        #   Spectral assertions on each synth engine
│   └── test_effects.py          #   Amplitude / behavioural assertions on each effect
├── samples/                     # User-supplied instrument samples for PSOLA
├── References/                  # Assignment artefacts and source notebooks
│   ├── TP2 - 25.20 ... Consigna.pdf    # assignment statement
│   ├── fft.c                            # Exercise 1.1 — Cooley-Tukey FFT in C
│   ├── Prueba_fft.exe                   # pre-compiled FFT testbench
│   ├── TP2 ASSD.ipynb                   # original PSOLA / Karplus notebook
│   ├── sound_effects_alumnos.ipynb      # original effects notebook
│   ├── additive_synthesis.py            # legacy additive script (now superseded by synthesis/additive.py)
│   ├── Concierto-De-Aranjuez.mid        # main test MIDI
│   ├── arpegio_didactico.mid            # tiny demo MIDI
│   ├── aranjuez_test_8s.wav             # sample render for verification
│   └── LEAME.txt
├── requirements.txt
└── README.md
```

---

## Installation

The app targets **Python 3.10+**. A virtual environment is recommended.

```bash
# 1. Clone
git clone <your-fork-url>
cd TP2-ASSD

# 2. Create and activate a venv
python3 -m venv venv
source venv/bin/activate            # Linux/macOS
# venv\Scripts\activate              # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:
```
PyQt6
numpy
scipy
pretty_midi
sounddevice
soundfile
pyqtgraph
pytest
```

On Linux you may also need ALSA / PortAudio headers for `sounddevice`:
```bash
sudo apt install libportaudio2
```

### Exercise 1.1 (C FFT)

The FFT in C is a standalone deliverable, kept in `References/`.

```bash
cd References
gcc -O2 fft.c -o fft -lm && ./fft
```

The program runs four testbenches (impulse, sinusoid, in-place, N=1) and prints the maximum error against the reference DFT.

---

## Usage

### Launching the GUI

```bash
python TP2_Espectro-MIDI.py
```

### Workflow

1. **Cargar MIDI…** — pick any `.mid` file. The track table is populated, one row per instrument in the MIDI file.
2. **Configure each track**:
   - **Canal**: editable channel number. Affects only the timeline row (audio routing is per-instrument). Reassigning two tracks to the same channel makes them share a row.
   - **Método**: synthesis algorithm — `(ninguno)` / `Karplus-Strong` / `Aditiva` / `FM` / `PSOLA`.
   - **Preset**: timbre options that update when the method changes.
     - Karplus-Strong → `String` / `Percussion`
     - Aditiva → `organ` / `piano` / `bell` / `flute`
     - FM → `clarinet` / `brass` / `bell` / `wood`
     - PSOLA → file picker for a WAV/AIF sample (the assumed pitch is C4; loaded file name appears in the cell)
   - **Nivel**: 1–5 (Aditiva only). Maps to PDF §2.3 accumulating improvements — set to compare perceptual quality at each tier.
   - **Gain**: per-track gain multiplier.
   - **Edit…**: opens the per-track effect chain editor.
3. **Edit Master Effects…** — opens a chain editor that runs after the mix is summed.
4. **Render (resíntesis)** — synthesizes the audio using each track's selected engine, applies per-track effects, mixes, then applies the master chain. The waveform, spectrogram and timeline panels all update.
5. **Reproducir / Detener** — plays the rendered audio.
6. **Exportar WAV…** — saves a normalized 16-bit WAV.

### Spectrogram parameters

The central panel exposes the three knobs the PDF calls out: window function (Hann / Hamming / Blackman / Boxcar), segment length `N`, and overlap percentage. The plot updates live whenever any value changes.

### Timeline visualizer

The bottom plot of the central column draws MIDI events FL Studio-style: one row per channel, each note as a horizontal rectangle whose extent is its duration. Each source instrument has a distinct colour, so reassigning a track's channel (via the Canal spinbox) is reflected immediately as the coloured bars shift up or down.

---

## Running the testbenches

```bash
pytest -q
```

20 tests run by default:
- Synthesis engines: amplitude sanity, fundamental-frequency / harmonic-series assertions, all additive levels execute, FM sidebands present.
- Effects: echo delays the signal by the expected sample offset, clipper respects its threshold, soft-clip is bounded, every modulation effect runs end-to-end, effect chain applies in order.

---

## Architecture notes

The codebase is structured around two narrow protocols so synth engines and effects are fully interchangeable.

```python
# synthesis/base.py
class SynthEngine(Protocol):
    name: str
    def synthesize_note(self, note: MidiNote, fs: int = 44100) -> np.ndarray: ...
    def render_track(self, notes: list[MidiNote], fs: int = 44100) -> np.ndarray: ...

# effects/base.py
class AudioEffect(Protocol):
    name: str
    def process(self, audio: np.ndarray, fs: int = 44100) -> np.ndarray: ...
    def reset_state(self) -> None: ...

class EffectChain:
    effects: list[AudioEffect]
    def process(self, audio, fs): ...
```

The `Mixer` (`audio/mixer.py`) walks the MIDI's `instruments`, converts each `pretty_midi.Note` to a `MidiNote`, calls the assigned synth, applies the per-track `EffectChain`, sums into a master buffer, and finally applies the master chain.

The DSP and the GUI are decoupled: each `synthesis/*.py` and `effects/*.py` module is importable with no Qt or notebook dependency.

### What is **not** library code

Per the PDF (page 2), only the following may use third-party libraries:
- MIDI decoding → `pretty_midi`
- Audio file I/O → `soundfile`, `scipy.io.wavfile`
- FFT for the spectrogram (1.2) → `scipy.signal.spectrogram`
- GUI / frontend → `PyQt6`, `pyqtgraph`, `sounddevice`

All synthesis, effects, PSOLA pitch detection, Karplus-Strong delay-line modeling, additive partial generation, FM phase modulation, and the 1.1 FFT itself are implemented from scratch in this project.

---

## Performance notes

- The Karplus-Strong inner loop is a pure-Python sample loop, which is the slowest path. Short MIDI files (under 30 s) render in well under a second per track at 22 050 Hz. A full *Concierto de Aranjuez* render with all 10 tracks on Karplus will take several minutes; for benchmarking you can lower `fs` or mix in faster engines (Additive, FM).
- Effects are mostly NumPy-vectorised, except the phaser, which iterates sample-by-sample to chase its LFO.
- For real-time playback the rendered audio is buffered in memory and sent to `sounddevice.play()` — there is no streaming engine.

---

## Credits

- Subject: **25.20 — Análisis de Señales y Sistemas Digitales**, ITBA, 1er cuatrimestre 2026.
- Lecturers (as listed in the assignment): Daniel Jacoby, Matías Bergerman, Bruno Di Sanzo, Nicolás Beade.
- Reference material:
  - J. O. Smith III — *Physical Audio Signal Processing* — https://ccrma.stanford.edu/~jos/pasp/
  - U. Zölzer (ed.) — *DAFX: Digital Audio Effects*, 2nd ed., Wiley, 2011.
  - Aalto University — *Virtual Analog Synthesis and Audio Effects*.

Per the assignment statement, all DSP, synthesis and effect algorithms are own implementations; third-party libraries are restricted to MIDI decoding, audio I/O, the spectrogram FFT, and GUI plumbing.
