"""FL Studio-style timeline visualizer for MIDI events.

One row per channel. Each note is drawn as a horizontal rectangle whose
horizontal extent is the note's duration. Bars are colour-coded by source
instrument so swaps remain visually traceable.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pretty_midi
import pyqtgraph as pg
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class TimelinePanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget(title="Timeline · eventos MIDI por canal")
        self.plot.setLabel("bottom", "Tiempo", units="s")
        self.plot.setLabel("left", "Canal")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.invertY(True)  # channel 0 at top, like FL playlist
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setMenuEnabled(False)
        layout.addWidget(self.plot)

        self._midi: Optional[pretty_midi.PrettyMIDI] = None
        self._channel_map: dict[int, int] = {}

    # public api
    def set_midi(self, midi: Optional[pretty_midi.PrettyMIDI], channel_map: dict[int, int]) -> None:
        self._midi = midi
        self._channel_map = dict(channel_map)
        self._redraw()

    def set_channel_map(self, channel_map: dict[int, int]) -> None:
        self._channel_map = dict(channel_map)
        self._redraw()

    def clear(self) -> None:
        self._midi = None
        self._channel_map = {}
        self.plot.clear()

    # internal
    def _redraw(self) -> None:
        self.plot.clear()
        if self._midi is None or not self._midi.instruments:
            return

        n_inst = len(self._midi.instruments)
        max_ch = 0
        min_ch = 0
        for inst_idx, instrument in enumerate(self._midi.instruments):
            channel = int(self._channel_map.get(inst_idx, inst_idx))
            if not instrument.notes:
                continue
            max_ch = max(max_ch, channel)
            min_ch = min(min_ch, channel)

            starts = np.array([n.start for n in instrument.notes], dtype=np.float64)
            durations = np.array([max(n.end - n.start, 0.005) for n in instrument.notes], dtype=np.float64)
            y0 = np.full(starts.shape, channel - 0.4)
            height = np.full(starts.shape, 0.8)

            color = pg.intColor(inst_idx, hues=max(8, n_inst))
            bars = pg.BarGraphItem(
                x0=starts, width=durations,
                y0=y0, height=height,
                brush=color, pen=pg.mkPen(color=(60, 60, 60), width=0.5),
            )
            self.plot.addItem(bars)

        self.plot.setYRange(min_ch - 0.6, max_ch + 0.6)
        # X range from 0 to track end
        total_end = self._midi.get_end_time()
        if total_end > 0:
            self.plot.setXRange(0, total_end, padding=0.02)
