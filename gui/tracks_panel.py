"""Per-track configuration panel.

Columns: Channel | Instrument | Method | Preset | Level (Additive only) | Gain | Effects.

- Channel: user-editable. Changing it only reorders the timeline; audio routing
  is unchanged (the mixer still keys tracks by source instrument index).
- Method / Preset / Level are split so the user can independently pick the
  synthesis algorithm, the instrument timbre, and the additive-quality level
  (matching PDF §2.3 a-d).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pretty_midi
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from audio.mixer import Mixer, TrackConfig
from effects.base import EffectChain
from synthesis import (
    KarplusStrongSynth, PSOLASynth, AdditiveSynth, ADDITIVE_PRESETS,
    FMSynth, FM_PRESETS,
)

from .effects_panel import EffectChainDialog
from .widgets import BUTTON_STYLE


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


# Synthesis method labels (the algorithm)
METHOD_NONE = "(ninguno)"
METHOD_KARPLUS = "Karplus-Strong"
METHOD_ADDITIVE = "Aditiva"
METHOD_FM = "FM"
METHOD_PSOLA = "PSOLA"
METHODS = [METHOD_NONE, METHOD_KARPLUS, METHOD_ADDITIVE, METHOD_FM, METHOD_PSOLA]

# Instrument presets per method (the timbre)
PRESETS_BY_METHOD: dict[str, list[str]] = {
    METHOD_NONE: [],
    METHOD_KARPLUS: ["String", "Percussion"],
    METHOD_ADDITIVE: list(ADDITIVE_PRESETS.keys()),  # organ, piano, bell, flute
    METHOD_FM: list(FM_PRESETS.keys()),               # clarinet, brass, bell, wood
    METHOD_PSOLA: [],  # populated as the loaded sample filename
}


# Column indexes
COL_CHANNEL = 0
COL_INSTRUMENT = 1
COL_METHOD = 2
COL_PRESET = 3
COL_LEVEL = 4
COL_GAIN = 5
COL_EFFECTS = 6


def build_synth(
    method: str,
    preset: Optional[str] = None,
    level: int = 3,
    sample_path: Optional[str] = None,
):
    if method == METHOD_NONE or not method:
        return None
    if method == METHOD_KARPLUS:
        if preset == "Percussion":
            return KarplusStrongSynth(RL=0.92, b=0.5, tail=0.2)
        return KarplusStrongSynth(RL=0.99, b=1.0, tail=0.5)
    if method == METHOD_ADDITIVE:
        return AdditiveSynth(preset=preset or "organ", level=int(level))
    if method == METHOD_FM:
        return FMSynth(preset=preset or "clarinet")
    if method == METHOD_PSOLA:
        if not sample_path:
            return None
        return PSOLASynth(sample_path=sample_path, base_midi=60)
    return None


class TracksPanel(QWidget):
    """Multi-track configuration. Owns the Mixer that the main window renders."""

    def __init__(self, on_channels_changed: Optional[Callable[[dict[int, int]], None]] = None):
        super().__init__()
        self.mixer = Mixer()
        self.on_channels_changed = on_channels_changed

        # Per-instrument display state (independent of mixer routing)
        self._inst_channels: dict[int, int] = {}     # inst_idx -> channel number (display only)
        self._sample_paths: dict[int, str] = {}      # inst_idx -> WAV/AIF path for PSOLA

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Tracks"))

        self.table = QTableWidget(0, 7, self)
        self.table.setHorizontalHeaderLabels(
            ["Canal", "Instrumento", "Método", "Preset", "Nivel", "Gain", "Efectos"]
        )
        # Set column widths
        widths = [55, 145, 130, 105, 60, 65, 75]
        for i, w in enumerate(widths):
            self.table.setColumnWidth(i, w)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        layout.addWidget(QLabel("Master Effects"))
        self.btn_master = QPushButton("Edit Master Effects…")
        self.btn_master.setStyleSheet(BUTTON_STYLE)
        self.btn_master.clicked.connect(self._edit_master)
        layout.addWidget(self.btn_master)

    # ---------- channel map / public ----------
    def get_channel_map(self) -> dict[int, int]:
        return dict(self._inst_channels)

    def populate_from_midi(self, midi: pretty_midi.PrettyMIDI) -> None:
        self.table.setRowCount(0)
        self.mixer = Mixer()
        self._inst_channels.clear()
        self._sample_paths.clear()

        for idx, instrument in enumerate(midi.instruments):
            self._add_row(idx, instrument)

        # Initialize mixer with default selections
        for idx in range(len(midi.instruments)):
            self._rebuild_synth(idx)

        if self.on_channels_changed:
            self.on_channels_changed(self.get_channel_map())

    # ---------- row construction ----------
    def _add_row(self, idx: int, instrument: pretty_midi.Instrument) -> None:
        self.table.insertRow(idx)

        # Channel
        ch_spin = QSpinBox()
        ch_spin.setRange(0, 99)
        ch_spin.setValue(idx)
        self._inst_channels[idx] = idx
        self.table.setCellWidget(idx, COL_CHANNEL, ch_spin)

        # Instrument label
        name = "Percussion" if instrument.is_drum else pretty_midi.program_to_instrument_name(instrument.program)
        item = QTableWidgetItem(f"{name} ({len(instrument.notes)})")
        self.table.setItem(idx, COL_INSTRUMENT, item)

        # Method
        method_combo = QComboBox()
        method_combo.addItems(METHODS)
        default_method = METHOD_KARPLUS
        method_combo.setCurrentText(default_method)
        self.table.setCellWidget(idx, COL_METHOD, method_combo)

        # Preset
        preset_combo = QComboBox()
        self.table.setCellWidget(idx, COL_PRESET, preset_combo)

        # Level (additive only)
        level_spin = QSpinBox()
        level_spin.setRange(1, 5)
        level_spin.setValue(3)
        level_spin.setToolTip("Calidad de la síntesis aditiva (1=estática, 5=exponencial por parcial)")
        self.table.setCellWidget(idx, COL_LEVEL, level_spin)

        # Gain
        gain_spin = QDoubleSpinBox()
        gain_spin.setRange(0.0, 4.0)
        gain_spin.setSingleStep(0.1)
        gain_spin.setValue(1.0)
        self.table.setCellWidget(idx, COL_GAIN, gain_spin)

        # Effects button
        btn = QPushButton("Edit…")
        btn.setStyleSheet(BUTTON_STYLE)
        btn.clicked.connect(lambda _=False, i=idx: self._edit_track_effects(i))
        self.table.setCellWidget(idx, COL_EFFECTS, btn)

        # Populate preset combo for current method, set level enabled state
        self._refresh_preset_combo(idx, default_method, default_preset=None)

        # Wire signals (after initial state setup, so we don't fire spuriously)
        ch_spin.valueChanged.connect(lambda v, i=idx: self._on_channel_changed(i, int(v)))
        method_combo.currentTextChanged.connect(lambda txt, i=idx: self._on_method_changed(i, txt))
        preset_combo.currentTextChanged.connect(lambda _txt, i=idx: self._rebuild_synth(i))
        level_spin.valueChanged.connect(lambda _v, i=idx: self._rebuild_synth(i))
        gain_spin.valueChanged.connect(lambda v, i=idx: self._on_gain_changed(i, float(v)))

    # ---------- per-row helpers ----------
    def _refresh_preset_combo(self, idx: int, method: str, default_preset: Optional[str]) -> None:
        preset_combo: QComboBox = self.table.cellWidget(idx, COL_PRESET)  # type: ignore[assignment]
        level_spin: QSpinBox = self.table.cellWidget(idx, COL_LEVEL)      # type: ignore[assignment]

        preset_combo.blockSignals(True)
        preset_combo.clear()
        items = PRESETS_BY_METHOD.get(method, [])
        if method == METHOD_PSOLA:
            sample_path = self._sample_paths.get(idx)
            label = Path(sample_path).name if sample_path else "(sin muestra)"
            preset_combo.addItem(label)
            preset_combo.setEnabled(False)
        else:
            preset_combo.addItems(items)
            preset_combo.setEnabled(bool(items))
            if default_preset and default_preset in items:
                preset_combo.setCurrentText(default_preset)
        preset_combo.blockSignals(False)

        # Level is meaningful only for Additive
        level_spin.setEnabled(method == METHOD_ADDITIVE)

    def _selected_for(self, idx: int) -> tuple[str, Optional[str], int]:
        method_combo: QComboBox = self.table.cellWidget(idx, COL_METHOD)   # type: ignore[assignment]
        preset_combo: QComboBox = self.table.cellWidget(idx, COL_PRESET)   # type: ignore[assignment]
        level_spin: QSpinBox = self.table.cellWidget(idx, COL_LEVEL)       # type: ignore[assignment]
        method = method_combo.currentText()
        preset = preset_combo.currentText() if preset_combo.isEnabled() else None
        level = int(level_spin.value())
        return method, preset, level

    def _rebuild_synth(self, idx: int) -> None:
        method, preset, level = self._selected_for(idx)
        sample_path = self._sample_paths.get(idx)
        try:
            synth = build_synth(method=method, preset=preset, level=level, sample_path=sample_path)
        except Exception as e:
            QMessageBox.warning(self, "Synth error", f"Failed to build synth: {e}")
            synth = None
        cfg = self.mixer.tracks.get(idx)
        if cfg is None:
            cfg = TrackConfig()
            self.mixer.set_track(idx, cfg)
        cfg.synth = synth

    # ---------- signal handlers ----------
    def _on_channel_changed(self, idx: int, value: int) -> None:
        self._inst_channels[idx] = int(value)
        if self.on_channels_changed:
            self.on_channels_changed(self.get_channel_map())

    def _on_method_changed(self, idx: int, method: str) -> None:
        if method == METHOD_PSOLA:
            path = self._prompt_sample(idx)
            if not path:
                # User cancelled — revert
                method_combo: QComboBox = self.table.cellWidget(idx, COL_METHOD)  # type: ignore[assignment]
                method_combo.blockSignals(True)
                method_combo.setCurrentText(METHOD_NONE)
                method_combo.blockSignals(False)
                self._refresh_preset_combo(idx, METHOD_NONE, default_preset=None)
                self._rebuild_synth(idx)
                return
            self._sample_paths[idx] = path
        self._refresh_preset_combo(idx, method, default_preset=None)
        self._rebuild_synth(idx)

    def _on_gain_changed(self, idx: int, value: float) -> None:
        cfg = self.mixer.tracks.get(idx)
        if cfg is not None:
            cfg.gain = value

    def _edit_track_effects(self, idx: int) -> None:
        cfg = self.mixer.tracks.get(idx)
        if cfg is None:
            return
        dlg = EffectChainDialog(cfg.effects, title=f"Track {idx} Effects", parent=self)
        dlg.exec()

    def _edit_master(self) -> None:
        dlg = EffectChainDialog(self.mixer.master_effects, title="Master Effects", parent=self)
        dlg.exec()

    def _prompt_sample(self, idx: int) -> Optional[str]:
        start_dir = str(SAMPLES_DIR) if SAMPLES_DIR.exists() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Cargar muestra para track {idx}", start_dir,
            "Audio (*.wav *.aif *.aiff *.flac)"
        )
        return path or None
