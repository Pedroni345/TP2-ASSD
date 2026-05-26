"""Effect chain editor dialog/widget."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)

from effects import EFFECT_REGISTRY, EffectChain
from .widgets import BUTTON_STYLE


class EffectChainEditor(QWidget):
    """A simple list editor: add/remove/reorder effects on a chain."""

    def __init__(self, chain: EffectChain, title: str = "Effect Chain"):
        super().__init__()
        self.chain = chain

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
        layout.addWidget(self.list)

        # Action buttons
        btns = QHBoxLayout()
        for label, fn in [("Up", self._move_up), ("Down", self._move_down), ("Remove", self._remove)]:
            b = QPushButton(label)
            b.setStyleSheet(BUTTON_STYLE)
            b.clicked.connect(fn)
            btns.addWidget(b)
        layout.addLayout(btns)

        self._refresh()

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
            # ConvolutionReverb requires a path; skip for now or surface error later
            if cls_name == "Conv. Reverb":
                from PyQt6.QtWidgets import QFileDialog, QMessageBox
                path, _ = QFileDialog.getOpenFileName(self, "Select IR file", "", "Audio (*.wav *.aif *.aiff)")
                if not path:
                    return
                eff = cls(ir_path=path)
            else:
                eff = cls()
            self.chain.add(eff)
            self._refresh()
        except Exception as e:
            print(f"[EffectChainEditor] failed to add {cls_name}: {e}")

    def _selected_idx(self) -> int:
        return self.list.currentRow()

    def _remove(self) -> None:
        idx = self._selected_idx()
        if idx >= 0:
            self.chain.remove(idx)
            self._refresh()

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


class EffectChainDialog(QDialog):
    def __init__(self, chain: EffectChain, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(360, 360)
        layout = QVBoxLayout(self)
        self.editor = EffectChainEditor(chain, title)
        layout.addWidget(self.editor)
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(BUTTON_STYLE)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
