"""TP2 ASSD - MIDI Synthesizer + Spectral Analysis main GUI.

Wires together synthesis engines (PSOLA, Karplus-Strong, Additive, FM),
per-track + master effect chains, configurable spectrogram, playback and
WAV export. Replaces the placeholder pretty_midi.synthesize() entirely.
"""
import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QSplitter,
    QLabel, QPushButton, QFileDialog, QMessageBox, QComboBox, QSpinBox,
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

try:
    import pretty_midi
    import pyqtgraph as pg
except ImportError:
    print("Por favor, instala: pip install pretty_midi pyqtgraph PyQt6 scipy sounddevice soundfile")
    sys.exit(1)

from analysis.spectrogram import compute_spectrogram
from audio.playback import AudioPlayer
from audio.export import export_wav
from gui.tracks_panel import TracksPanel
from gui.timeline_panel import TimelinePanel
from gui.widgets import styled_group, BUTTON_STYLE


class SintetizadorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sintetizador Musical - Analisis Espectral (TP2 ASSD)")
        self.resize(1600, 900)
        self.setFont(QFont("Segoe UI", 11))

        self.midi_data: pretty_midi.PrettyMIDI | None = None
        self.audio: np.ndarray | None = None
        self.fs = 44100
        self.player = AudioPlayer()

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._left_panel())
        splitter.addWidget(self._center_panel())
        splitter.addWidget(self._right_panel())
        splitter.setSizes([220, 700, 660])
        root_layout.addWidget(splitter)

    # ----- panels -----
    def _left_panel(self):
        grp = styled_group("MIDI / ARCHIVOS", "#2196F3")
        layout = QVBoxLayout(grp)

        self.btn_cargar = QPushButton("Cargar MIDI…")
        self.btn_render = QPushButton("Render (resíntesis)")
        self.btn_play = QPushButton("Reproducir")
        self.btn_export = QPushButton("Exportar WAV…")
        for b in (self.btn_cargar, self.btn_render, self.btn_play, self.btn_export):
            b.setStyleSheet(BUTTON_STYLE)
        self.btn_render.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_export.setEnabled(False)

        self.btn_cargar.clicked.connect(self._cargar_midi)
        self.btn_render.clicked.connect(self._render)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_export.clicked.connect(self._export)

        layout.addWidget(self.btn_cargar)
        layout.addWidget(self.btn_render)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_export)

        self.status = QLabel("Sin MIDI cargado.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        layout.addStretch()
        return grp

    def _center_panel(self):
        grp = styled_group("ANALISIS VISUAL", "#4CAF50")
        layout = QVBoxLayout(grp)

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Ventana:"))
        self.combo_ventana = QComboBox()
        self.combo_ventana.addItems(["hann", "hamming", "boxcar", "blackman"])
        controls.addWidget(self.combo_ventana)
        controls.addWidget(QLabel("Segmento (N):"))
        self.combo_nperseg = QComboBox()
        self.combo_nperseg.addItems(["256", "512", "1024", "2048", "4096"])
        self.combo_nperseg.setCurrentText("1024")
        controls.addWidget(self.combo_nperseg)
        controls.addWidget(QLabel("Solapamiento (%):"))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 99)
        self.spin_overlap.setValue(50)
        controls.addWidget(self.spin_overlap)
        layout.addLayout(controls)
        for w in (self.combo_ventana, self.combo_nperseg, self.spin_overlap):
            sig = getattr(w, "currentTextChanged", None) or getattr(w, "valueChanged", None)
            sig.connect(self._refresh_plots)

        self.plot_wave = pg.PlotWidget(title="Forma de Onda")
        self.plot_wave.setLabel("left", "Amplitud")
        self.plot_wave.setLabel("bottom", "Tiempo", units="s")
        self.plot_wave.showGrid(x=True, y=True, alpha=0.3)
        self.curve_wave = self.plot_wave.plot(pen=pg.mkPen(color="#2196F3", width=1.2))

        self.plot_spec = pg.PlotWidget(title="Espectrograma")
        self.plot_spec.setLabel("left", "Frecuencia", units="Hz")
        self.plot_spec.setLabel("bottom", "Tiempo", units="s")
        self.img_spec = pg.ImageItem()
        self.plot_spec.addItem(self.img_spec)
        self.img_spec.setColorMap(pg.colormap.get("inferno"))

        self.timeline_panel = TimelinePanel()

        layout.addWidget(self.plot_wave, stretch=2)
        layout.addWidget(self.plot_spec, stretch=3)
        layout.addWidget(self.timeline_panel, stretch=3)
        return grp

    def _right_panel(self):
        grp = styled_group("SINTESIS / EFECTOS", "#9C27B0")
        layout = QVBoxLayout(grp)
        self.tracks_panel = TracksPanel(on_channels_changed=self._on_channels_changed)
        layout.addWidget(self.tracks_panel)
        return grp

    def _on_channels_changed(self, channel_map: dict[int, int]) -> None:
        # Called by TracksPanel whenever a row's channel spinbox changes.
        self.timeline_panel.set_channel_map(channel_map)

    # ----- actions -----
    def _cargar_midi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo MIDI", "", "Archivos MIDI (*.mid *.midi)"
        )
        if not path:
            return
        try:
            if self.player.is_playing:
                self.player.stop()
                self.btn_play.setText("Reproducir")
            self.midi_data = pretty_midi.PrettyMIDI(path)
            self.tracks_panel.populate_from_midi(self.midi_data)
            self.timeline_panel.set_midi(self.midi_data, self.tracks_panel.get_channel_map())
            self.status.setText(
                f"MIDI cargado: {path.split('/')[-1]}\n"
                f"Duración: {self.midi_data.get_end_time():.1f}s · "
                f"Tracks: {len(self.midi_data.instruments)}"
            )
            self.btn_render.setEnabled(True)
            self.btn_play.setEnabled(False)
            self.btn_export.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fallo al cargar MIDI:\n{e}")

    def _render(self):
        if self.midi_data is None:
            return
        try:
            self.status.setText("Renderizando audio… (puede tardar)")
            QApplication.processEvents()
            audio = self.tracks_panel.mixer.render(self.midi_data, fs=self.fs)
            if audio.size == 0:
                QMessageBox.warning(self, "Render vacío", "No se generó audio. ¿Asignaste un synth a alguna pista?")
                return
            peak = float(np.max(np.abs(audio)))
            if peak > 0:
                audio = audio / peak
            self.audio = audio.astype(np.float32)
            self.btn_play.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.status.setText(
                f"Audio listo · {len(self.audio)/self.fs:.1f}s @ {self.fs} Hz"
            )
            self._refresh_plots()
        except Exception as e:
            QMessageBox.critical(self, "Error de render", f"{type(e).__name__}: {e}")
            self.status.setText("Render falló — ver consola.")
            import traceback; traceback.print_exc()

    def _refresh_plots(self):
        if self.audio is None or self.audio.size == 0:
            return
        t = np.arange(len(self.audio)) / self.fs
        step = max(1, len(self.audio) // 8000)
        self.curve_wave.setData(t[::step], self.audio[::step])

        window = self.combo_ventana.currentText()
        nperseg = int(self.combo_nperseg.currentText())
        overlap = self.spin_overlap.value() / 100.0
        try:
            f, tt, Sxx_db = compute_spectrogram(
                self.audio, self.fs, window=window, nperseg=nperseg, overlap_pct=overlap
            )
            self.img_spec.setImage(Sxx_db.T, autoLevels=True)
            rect = pg.QtCore.QRectF(0, 0, float(tt[-1]) if len(tt) else 1.0,
                                    float(f[-1]) if len(f) else self.fs / 2)
            self.img_spec.setRect(rect)
        except Exception as e:
            print(f"[spectrogram] {e}")

    def _toggle_play(self):
        if self.audio is None:
            return
        if not self.player.is_playing:
            self.player.play(self.audio, self.fs)
            self.btn_play.setText("Detener")
        else:
            self.player.stop()
            self.btn_play.setText("Reproducir")

    def _export(self):
        if self.audio is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar audio", "render.wav", "WAV (*.wav)")
        if not path:
            return
        try:
            export_wav(self.audio, self.fs, path)
            QMessageBox.information(self, "Exportar", f"Guardado en:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo exportar:\n{e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SintetizadorGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
