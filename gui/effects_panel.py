"""Effect chain editor dialog/widget.

Each effect class declares a ``PARAMS`` list describing its tunable knobs;
the panel introspects it to build a parameter editor on the fly.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from effects import EFFECT_REGISTRY, EffectChain
from .widgets import BUTTON_STYLE


def _build_param_editor(effect, parent: QWidget) -> QWidget:
    """Return a small widget with one spinbox per effect.PARAMS entry."""
    box = QWidget(parent)
    form = QFormLayout(box)
    form.setContentsMargins(4, 4, 4, 4)

    params = getattr(effect, "PARAMS", None)
    if not params:
        form.addRow(QLabel("(sin parámetros editables)"))
        return box

    for spec in params:
        attr, lo, hi, step, label = spec
        current = getattr(effect, attr)
        # Integer step => QSpinBox; otherwise QDoubleSpinBox.
        is_int = isinstance(step, int) and float(step).is_integer() and isinstance(current, int)
        if is_int:
            spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            spin.setRange(int(lo), int(hi))
            spin.setSingleStep(int(step))
            spin.setValue(int(current))
            spin.valueChanged.connect(
                lambda v, e=effect, a=attr: setattr(e, a, int(v))
            )
        else:
            spin = QDoubleSpinBox()
            spin.setRange(float(lo), float(hi))
            spin.setSingleStep(float(step))
            # Use enough decimals to actually display the step.
            decimals = 4 if float(step) < 0.005 else (3 if float(step) < 0.05 else 2)
            spin.setDecimals(decimals)
            spin.setValue(float(current))
            spin.valueChanged.connect(
                lambda v, e=effect, a=attr: setattr(e, a, float(v))
            )
        form.addRow(label, spin)
    return box


class EffectChainEditor(QWidget):
    """List of effects + per-effect parameter knobs."""

    def __init__(self, chain: EffectChain, title: str = "Effect Chain"):
        super().__init__()
        self.chain = chain
        self._params_box: QWidget | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(title))

        # Add effect row
        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems(EFFECT_REGISTRY.keys())
        btn_add = QPushButton("Add")
        btn_add.setStyleSheet(BUTTON_STYLE)
        btn_add.clicked.connect(self._add_effect)
        row.addWidget(self.combo)
        row.addWidget(btn_add)
        layout.addLayout(row)

        # Current chain list
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list, stretch=1)

        # Action buttons
        btns = QHBoxLayout()
        for label, fn in [("Up", self._move_up), ("Down", self._move_down), ("Remove", self._remove)]:
            b = QPushButton(label)
            b.setStyleSheet(BUTTON_STYLE)
            b.clicked.connect(fn)
            btns.addWidget(b)
        layout.addLayout(btns)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Params header + placeholder; replaced when selection changes.
        layout.addWidget(QLabel("Parámetros del efecto seleccionado"))
        self._params_container = QVBoxLayout()
        layout.addLayout(self._params_container, stretch=1)
        self._set_params_widget(QLabel("(seleccioná un efecto)"))

        self._refresh()

    # ---------- chain list ops ----------
    def _refresh(self) -> None:
        self.list.clear()
        for eff in self.chain.effects:
            item = QListWidgetItem(getattr(eff, "name", type(eff).__name__))
            self.list.addItem(item)

    def _add_effect(self) -> None:
        cls_name = self.combo.currentText()
        cls = EFFECT_REGISTRY.get(cls_name)
        if cls is None:
            return
        try:
            if cls_name == "Conv. Reverb":
                from PyQt6.QtWidgets import QFileDialog
                path, _ = QFileDialog.getOpenFileName(self, "Select IR file", "", "Audio (*.wav *.aif *.aiff)")
                if not path:
                    return
                eff = cls(ir_path=path)
            else:
                eff = cls()
            self.chain.add(eff)
            self._refresh()
            self.list.setCurrentRow(len(self.chain.effects) - 1)
        except Exception as e:
            print(f"[EffectChainEditor] failed to add {cls_name}: {e}")

    def _selected_idx(self) -> int:
        return self.list.currentRow()

    def _remove(self) -> None:
        idx = self._selected_idx()
        if idx >= 0:
            self.chain.remove(idx)
            self._refresh()
            self._set_params_widget(QLabel("(seleccioná un efecto)"))

    def _move_up(self) -> None:
        idx = self._selected_idx()
        if idx > 0:
            self.chain.effects[idx - 1], self.chain.effects[idx] = self.chain.effects[idx], self.chain.effects[idx - 1]
            self._refresh()
            self.list.setCurrentRow(idx - 1)

    def _move_down(self) -> None:
        idx = self._selected_idx()
        if 0 <= idx < len(self.chain.effects) - 1:
            self.chain.effects[idx + 1], self.chain.effects[idx] = self.chain.effects[idx], self.chain.effects[idx + 1]
            self._refresh()
            self.list.setCurrentRow(idx + 1)

    # ---------- params editor ----------
    def _on_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.chain.effects):
            self._set_params_widget(QLabel("(seleccioná un efecto)"))
            return
        eff = self.chain.effects[row]
        self._set_params_widget(_build_param_editor(eff, self))

    def _set_params_widget(self, widget: QWidget) -> None:
        if self._params_box is not None:
            self._params_container.removeWidget(self._params_box)
            self._params_box.deleteLater()
        self._params_box = widget
        self._params_container.addWidget(self._params_box)


class EffectChainDialog(QDialog):
    def __init__(self, chain: EffectChain, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 520)
        layout = QVBoxLayout(self)
        self.editor = EffectChainEditor(chain, title)
        layout.addWidget(self.editor)
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(BUTTON_STYLE)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
