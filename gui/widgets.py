"""Reusable styled widgets for the GUI."""
from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QGroupBox


def styled_group(title: str, color: str = "#2196F3") -> QGroupBox:
    grupo = QGroupBox(title)
    grupo.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
    grupo.setStyleSheet(f"""
        QGroupBox {{
            background-color: #f8f9fa;
            border: 2px solid #cccccc;
            border-radius: 8px;
            margin-top: 16px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top center;
            background-color: {color};
            color: white;
            padding: 5px 15px;
            border-radius: 6px;
        }}
    """)
    return grupo


BUTTON_STYLE = """
    QPushButton {
        padding: 6px; border-radius: 4px;
        background-color: #e0e0e0; border: 1px solid #cccccc;
    }
    QPushButton:hover { background-color: #d0d0d0; }
    QPushButton:disabled { background-color: #f5f5f5; color: #aaaaaa; }
"""
