"""
VerifAI — İleri Seviye Adli Bilişim Arayüzü
Güven skoru göstergesi, radar grafiği, algoritma detayları ve harita görüntüleyici.
"""

import numpy as np
import io

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QComboBox,
    QTabWidget, QScrollArea, QFrame, QProgressBar, QSizePolicy, QApplication
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QPen, QLinearGradient, QConicalGradient, QBrush
from PyQt5.QtCore import Qt, QTimer, QSize

from detector import ManipulationDetector
from confidence import ConfidenceEngine

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


# ─────────────────────────────────────────────
# RENK PALETİ
# ─────────────────────────────────────────────
COLORS = {
    "bg_dark":       "#0d1117",
    "bg_card":       "#161b22",
    "bg_card_hover": "#1c2333",
    "border":        "#30363d",
    "text_primary":  "#e6edf3",
    "text_secondary":"#8b949e",
    "accent_green":  "#2ecc71",
    "accent_blue":   "#3498db",
    "accent_yellow": "#f1c40f",
    "accent_orange": "#e67e22",
    "accent_red":    "#e74c3c",
    "accent_purple": "#8e44ad",
    "gradient_start":"#667eea",
    "gradient_end":  "#764ba2",
}

GLOBAL_STYLE = f"""
    QWidget {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_primary']};
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
    }}
    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        background: {COLORS['bg_dark']};
    }}
    QTabBar::tab {{
        background: {COLORS['bg_card']};
        color: {COLORS['text_secondary']};
        padding: 10px 24px;
        margin-right: 2px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        font-weight: bold;
        font-size: 13px;
    }}
    QTabBar::tab:selected {{
        background: {COLORS['gradient_start']};
        color: white;
    }}
    QTabBar::tab:hover {{
        background: {COLORS['bg_card_hover']};
        color: {COLORS['text_primary']};
    }}
    QPushButton {{
        border: none;
        border-radius: 10px;
        padding: 14px 28px;
        font-weight: bold;
        font-size: 14px;
    }}
    QPushButton:hover {{
        opacity: 0.9;
    }}
    QComboBox {{
        background: {COLORS['bg_card']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 13px;
    }}
    QComboBox:hover {{
        border-color: {COLORS['gradient_start']};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 30px;
    }}
    QComboBox QAbstractItemView {{
        background: {COLORS['bg_card']};
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['gradient_start']};
        border: 1px solid {COLORS['border']};
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        background: {COLORS['border']};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {COLORS['gradient_start']};
        width: 18px;
        height: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }}
    QSlider::sub-page:horizontal {{
        background: {COLORS['gradient_start']};
        border-radius: 3px;
    }}
    QScrollArea {{
        border: none;
    }}
    QScrollBar:vertical {{
        background: {COLORS['bg_dark']};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {COLORS['border']};
        border-radius: 4px;
        min-height: 30px;
    }}
"""


class UIManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VerifAI - AI Tespit Sistemi")
        self.resize(800, 600)
        self.setStyleSheet(GLOBAL_STYLE)
        layout = QVBoxLayout(self)
        label = QLabel("VerifAI - Analiz Paneli")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"font-size: 24px; color: {COLORS['gradient_start']};")
        layout.addWidget(label)
