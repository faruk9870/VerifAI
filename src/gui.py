"""
VerifAI — AI Gorsel Tespit Sistemi
Guven skoru gostergesi, radar grafigi, algoritma detaylari ve harita goruntuleyici.
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
# RENK PALETI
# ─────────────────────────────────────────────
COLORS = {
    "bg_dark":       "#ffffff",
    "bg_card":       "#f8f9fc",
    "bg_card_hover": "#eef1f8",
    "border":        "#e2e6ef",
    "text_primary":  "#1a1d2e",
    "text_secondary":"#6b7280",
    "accent_green":  "#10b981",
    "accent_blue":   "#3b82f6",
    "accent_yellow": "#f59e0b",
    "accent_orange": "#f97316",
    "accent_red":    "#ef4444",
    "accent_purple": "#8b5cf6",
    "gradient_start":"#6366f1",
    "gradient_end":  "#8b5cf6",
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
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLORS['gradient_start']}, stop:1 {COLORS['gradient_end']});
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


class ScoreGaugeWidget(QWidget):
    """Yuvarlak guven skoru gostergesi widget'i."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.score = 0
        self.verdict = ""
        self.color = COLORS['text_secondary']
        self.emoji = ""
        self.setMinimumSize(220, 220)
        self.setMaximumSize(260, 260)

    def set_score(self, score, verdict, color, emoji):
        self.score = score
        self.verdict = verdict
        self.color = color
        self.emoji = emoji
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        center_x = w // 2
        center_y = h // 2
        radius = min(w, h) // 2 - 20

        # Arka plan halkasi
        pen = QPen(QColor(COLORS['border']), 12)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(
            center_x - radius, center_y - radius,
            radius * 2, radius * 2,
            225 * 16, -270 * 16
        )

        # Skor halkasi
        pen = QPen(QColor(self.color), 12)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        angle = int(-270 * (self.score / 100.0))
        painter.drawArc(
            center_x - radius, center_y - radius,
            radius * 2, radius * 2,
            225 * 16, angle * 16
        )

        # Skor metni
        painter.setPen(QColor(self.color))
        font = QFont("Segoe UI", 36, QFont.Bold)
        painter.setFont(font)
        painter.drawText(
            center_x - radius, center_y - 30,
            radius * 2, 50,
            Qt.AlignCenter, f"{self.score}"
        )

        # Karar metni
        painter.setPen(QColor(COLORS['text_secondary']))
        font = QFont("Segoe UI", 11, QFont.Bold)
        painter.setFont(font)
        painter.drawText(
            center_x - radius, center_y + 20,
            radius * 2, 30,
            Qt.AlignCenter, self.verdict
        )

        painter.end()


class AlgorithmCard(QFrame):
    """Tek bir algoritmanin sonucunu gosteren kart widget'i."""

    def __init__(self, name, display_name, description, score, active=True, parent=None):
        super().__init__(parent)
        # Kalkan sistemi float uretebilir, PyQt int bekler
        score = int(round(score))
        
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                padding: 12px;
            }}
            QFrame:hover {{
                border-color: {COLORS['gradient_start']};
                background: {COLORS['bg_card_hover']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Ust satir: Isim + Skor
        top_row = QHBoxLayout()

        lbl_name = QLabel(f"  {display_name}")
        lbl_name.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold; font-size: 13px; border: none;")
        top_row.addWidget(lbl_name)

        top_row.addStretch()

        color = self._score_color(score, active)
        lbl_score = QLabel(f"{score}/100" if active else "N/A")
        lbl_score.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 15px; border: none;")
        top_row.addWidget(lbl_score)

        layout.addLayout(top_row)

        # Skor cubugu
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(score if active else 0)
        progress.setTextVisible(False)
        progress.setFixedHeight(6)
        progress.setStyleSheet(f"""
            QProgressBar {{
                background: {COLORS['border']};
                border-radius: 3px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(progress)

        # Aciklama
        lbl_desc = QLabel(description)
        lbl_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; border: none;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        if not active:
            inactive_lbl = QLabel("⏸ Devre dışı (Sosyal medya sıkıştırması)")
            inactive_lbl.setStyleSheet(f"color: {COLORS['accent_orange']}; font-size: 10px; font-style: italic; border: none;")
            layout.addWidget(inactive_lbl)

    @staticmethod
    def _score_color(score, active):
        if not active:
            return COLORS['text_secondary']
        if score < 25:
            return COLORS['accent_green']
        elif score < 45:
            return COLORS['accent_blue']
        elif score < 65:
            return COLORS['accent_yellow']
        elif score < 85:
            return COLORS['accent_orange']
        else:
            return COLORS['accent_red']


class UIManager(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.current_maps = {}
        self.last_result = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("VerifAI — AI Gorsel Tespit Sistemi")
        self.resize(1280, 850)
        self.setStyleSheet(GLOBAL_STYLE)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)

        # ─── HEADER ─────────────────────────
        header = QHBoxLayout()

        title_label = QLabel("🔬 VerifAI")
        title_label.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            color: {COLORS['gradient_start']};
            padding: 4px;
        """)
        header.addWidget(title_label)

        subtitle = QLabel("AI Gorsel Tespit Sistemi")
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; padding-top: 10px; font-weight: 500;")
        header.addWidget(subtitle)
        header.addStretch()

        main_layout.addLayout(header)

        # ─── KONTROL CUBUGU ─────────────────
        controls = QHBoxLayout()

        self.btn_load = QPushButton("📂  Fotoğraf Yükle")
        self.btn_load.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['bg_card']}, stop:1 {COLORS['bg_card_hover']});
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
            }}
            QPushButton:hover {{
                border-color: {COLORS['gradient_start']};
                background: {COLORS['bg_card_hover']};
            }}
        """)
        self.btn_load.setMinimumHeight(48)
        self.btn_load.clicked.connect(self.load_image)
        controls.addWidget(self.btn_load)

        self.btn_analyze = QPushButton("🧠  Analizi Başlat")
        self.btn_analyze.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['gradient_start']}, stop:1 {COLORS['gradient_end']});
                color: white;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['gradient_end']}, stop:1 {COLORS['gradient_start']});
            }}
            QPushButton:disabled {{
                background: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.btn_analyze.setMinimumHeight(48)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_analyze.setEnabled(False)
        controls.addWidget(self.btn_analyze)

        # FFT esik degeri varsayilan olarak 15 kullanilir
        self.slider = type('obj', (object,), {'value': lambda self: 15})()

        controls.addStretch()
        main_layout.addLayout(controls)

        # ─── ANA ICERIK: TAB SISTEMI ──────
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)

        # TAB 1: Analiz Sonuclari
        self.tab_results = QWidget()
        self.init_results_tab()
        self.tabs.addTab(self.tab_results, "📊 Analiz Sonuçları")

        # TAB 2: Gorsel Haritalar
        self.tab_maps = QWidget()
        self.init_maps_tab()
        self.tabs.addTab(self.tab_maps, "🗺️ Görsel Haritalar")

    def init_results_tab(self):
        """Analiz sonuclari tab'ini olusturur."""
        layout = QHBoxLayout(self.tab_results)
        layout.setSpacing(16)
        layout.setContentsMargins(12, 12, 12, 12)

        # Sol taraf: Orijinal gorsel + Gauge
        left_panel = QVBoxLayout()

        # Orijinal gorsel
        self.lbl_original = QLabel("Bir fotoğraf yükleyin")
        self.lbl_original.setAlignment(Qt.AlignCenter)
        self.lbl_original.setStyleSheet(f"""
            border: 2px dashed {COLORS['border']};
            border-radius: 12px;
            background: {COLORS['bg_card']};
            color: {COLORS['text_secondary']};
            font-size: 14px;
            min-height: 400px;
        """)
        self.lbl_original.setMinimumSize(520, 420)
        left_panel.addWidget(self.lbl_original)

        # Gauge
        gauge_container = QHBoxLayout()
        gauge_container.addStretch()
        self.gauge = ScoreGaugeWidget()
        self.gauge.set_score(0, "Bekleniyor", COLORS['text_secondary'], "")
        gauge_container.addWidget(self.gauge)
        gauge_container.addStretch()
        left_panel.addLayout(gauge_container)

        # Sonuc aciklama
        self.lbl_verdict_desc = QLabel("")
        self.lbl_verdict_desc.setAlignment(Qt.AlignCenter)
        self.lbl_verdict_desc.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
            padding: 8px;
        """)
        self.lbl_verdict_desc.setWordWrap(True)
        left_panel.addWidget(self.lbl_verdict_desc)

        left_panel.addStretch()
        layout.addLayout(left_panel, 4)

        # Sag taraf: Algoritma kartlari + Radar
        right_panel = QVBoxLayout()

        # Radar chart placeholder
        self.lbl_radar = QLabel("Radar grafiği analiz sonrası görünecek")
        self.lbl_radar.setAlignment(Qt.AlignCenter)
        self.lbl_radar.setStyleSheet(f"""
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            background: {COLORS['bg_card']};
            color: {COLORS['text_secondary']};
            font-size: 12px;
        """)
        self.lbl_radar.setMinimumSize(350, 280)
        self.lbl_radar.setMaximumHeight(320)
        right_panel.addWidget(self.lbl_radar)

        # Algoritma kartlari scroll alani
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background: transparent;")

        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setSpacing(8)
        self.cards_layout.setContentsMargins(4, 4, 4, 4)

        # Baslangicta bos kart alani
        placeholder = QLabel("Algoritma sonuçları analiz sonrası burada görünecek")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; padding: 30px;")
        self.cards_layout.addWidget(placeholder)
        self.cards_layout.addStretch()

        scroll.setWidget(self.cards_container)
        right_panel.addWidget(scroll, 1)

        layout.addLayout(right_panel, 5)

    def init_maps_tab(self):
        """Gorsel haritalar tab'ini olusturur."""
        layout = QVBoxLayout(self.tab_maps)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # Harita secici
        top_bar = QHBoxLayout()

        lbl = QLabel("Analiz Katmanı:")
        lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: bold; font-size: 13px;")
        top_bar.addWidget(lbl)

        self.combo_view = QComboBox()
        self.combo_view.addItems(["Analiz Bekleniyor..."])
        self.combo_view.setEnabled(False)
        self.combo_view.setMinimumWidth(250)
        self.combo_view.currentIndexChanged.connect(self.change_view)
        top_bar.addWidget(self.combo_view)
        top_bar.addStretch()

        layout.addLayout(top_bar)

        # Harita goruntuleme alani (yan yana: orijinal + analiz)
        map_area = QHBoxLayout()

        # Orijinal (kucuk)
        orig_container = QVBoxLayout()
        orig_label = QLabel("Orijinal")
        orig_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: bold; font-size: 12px;")
        orig_label.setAlignment(Qt.AlignCenter)
        orig_container.addWidget(orig_label)

        self.lbl_map_original = QLabel()
        self.lbl_map_original.setAlignment(Qt.AlignCenter)
        self.lbl_map_original.setStyleSheet(f"""
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            background: {COLORS['bg_card']};
        """)
        self.lbl_map_original.setMinimumSize(550, 480)
        orig_container.addWidget(self.lbl_map_original)
        map_area.addLayout(orig_container)

        # Analiz haritasi
        analysis_container = QVBoxLayout()
        self.lbl_map_title = QLabel("Analiz Haritası")
        self.lbl_map_title.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: bold; font-size: 12px;")
        self.lbl_map_title.setAlignment(Qt.AlignCenter)
        analysis_container.addWidget(self.lbl_map_title)

        self.lbl_map_result = QLabel("Sonuçlar burada görünecek")
        self.lbl_map_result.setAlignment(Qt.AlignCenter)
        self.lbl_map_result.setStyleSheet(f"""
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            background: {COLORS['bg_card']};
            color: {COLORS['text_secondary']};
        """)
        self.lbl_map_result.setMinimumSize(550, 480)
        analysis_container.addWidget(self.lbl_map_result)
        map_area.addLayout(analysis_container)

        layout.addLayout(map_area, 1)

    # ─── KONTROL FONKSIYONLARI ──────────────

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Analiz İçin Görsel Seç", "",
            "Resim Dosyaları (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path)

            # Tab 1: Orijinal
            self.lbl_original.setPixmap(
                pixmap.scaled(520, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.lbl_original.setStyleSheet(f"""
                border: 2px solid {COLORS['gradient_start']};
                border-radius: 12px;
                background: {COLORS['bg_card']};
            """)

            # Tab 2: Harita sayfasindaki orijinal
            self.lbl_map_original.setPixmap(
                pixmap.scaled(550, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            # Resetle
            self.gauge.set_score(0, "Hazır", COLORS['gradient_start'], "")
            self.lbl_verdict_desc.setText("Analiz başlatmak için 🧠 butonuna tıklayın")
            self.combo_view.clear()
            self.combo_view.addItem("Lütfen Analizi Başlatın")
            self.combo_view.setEnabled(False)
            self.btn_analyze.setEnabled(True)

            # Kartlari temizle
            self._clear_cards()

    def run_analysis(self):
        if not self.image_path:
            return

        # Analiz basliyor bildirimi
        self.gauge.set_score(0, "Analiz ediliyor...", COLORS['accent_yellow'], "⏳")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setText("⏳ Analiz Devam Ediyor...")
        QApplication.processEvents()

        try:
            threshold = self.slider.value()

            # Tum algoritmalari calistir
            result = ManipulationDetector.run_full_suite(self.image_path, threshold)

            # Guven skorunu hesapla
            confidence_result = ConfidenceEngine.compute(
                result["algorithm_scores"],
                result["modifiers"]
            )

            self.last_result = {**result, "confidence": confidence_result}
            self.current_maps = result["maps"]

            # UI'i guncelle
            self._update_gauge(confidence_result)
            self._update_cards(confidence_result, result)
            self._update_radar(result["algorithm_scores"], confidence_result)
            self._update_maps_tab()

        except Exception as e:
            self.gauge.set_score(0, "HATA", COLORS['accent_red'], "❌")
            self.lbl_verdict_desc.setText(f"Analiz hatası: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            self.btn_analyze.setEnabled(True)
            self.btn_analyze.setText("🧠  Analizi Başlat")

    # ─── UI GUNCELLEME ──────────────────────

    def _update_gauge(self, conf):
        self.gauge.set_score(
            int(conf["final_score"]),
            conf["verdict"],
            conf["color"],
            conf["emoji"]
        )

        desc = conf["description"]
        if conf.get("exif_adjustment", 0) != 0:
            desc += f"\n(EXIF duzeltmesi: {conf['exif_adjustment']:+d} puan)"
        self.lbl_verdict_desc.setText(desc)
        self.lbl_verdict_desc.setStyleSheet(f"""
            color: {conf['color']};
            font-size: 13px;
            font-weight: bold;
            padding: 8px;
        """)

    def _update_cards(self, conf, result):
        self._clear_cards()

        display_names = ConfidenceEngine.get_algo_display_names()
        descriptions = ConfidenceEngine.get_algo_descriptions()

        for algo_info in conf["per_algorithm"]:
            name = algo_info["name"]
            card = AlgorithmCard(
                name=name,
                display_name=display_names.get(name, name),
                description=descriptions.get(name, ""),
                score=algo_info["score"],
                active=algo_info["active"]
            )
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def _clear_cards(self):
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.cards_layout.addStretch()

    def _update_radar(self, scores, conf):
        """Matplotlib ile radar chart cizip QLabel'e yerlestirir."""
        display_names = ConfidenceEngine.get_algo_display_names()

        labels = []
        values = []
        for name, display in display_names.items():
            labels.append(display)
            values.append(scores.get(name, 0))

        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4, 3.5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(COLORS['bg_card'])
        ax.set_facecolor(COLORS['bg_card'])

        # Radar cizimi
        ax.fill(angles, values_plot, color=conf['color'], alpha=0.15)
        ax.plot(angles, values_plot, color=conf['color'], linewidth=2, marker='o', markersize=4)

        # Etiketler
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=7, color=COLORS['text_secondary'])
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color=COLORS['text_secondary'])
        ax.spines['polar'].set_color(COLORS['border'])
        ax.grid(color=COLORS['border'], alpha=0.3)
        ax.tick_params(colors=COLORS['text_secondary'])

        plt.tight_layout()

        # Figure → QPixmap
        pixmap = self._fig_to_pixmap(fig)
        plt.close(fig)

        self.lbl_radar.setPixmap(pixmap.scaled(
            self.lbl_radar.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def _update_maps_tab(self):
        self.combo_view.clear()
        self.combo_view.addItems(list(self.current_maps.keys()))
        self.combo_view.setEnabled(True)
        self.change_view()

    def change_view(self):
        """Harita secimi degistiginde."""
        selected = self.combo_view.currentText()
        if selected in self.current_maps:
            img_data = self.current_maps[selected]
            self.lbl_map_title.setText(f"📊 {selected}")

            if len(img_data.shape) == 2:
                h, w = img_data.shape
                # Renkli isi haritasi olustur
                colored = cv2.applyColorMap(img_data, cv2.COLORMAP_JET)
                colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                q_img = QImage(colored_rgb.data, w, h, w * 3, QImage.Format_RGB888)
            else:
                h, w, ch = img_data.shape
                q_img = QImage(img_data.data, w, h, w * ch, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            self.lbl_map_result.setPixmap(
                pixmap.scaled(550, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    @staticmethod
    def _fig_to_pixmap(fig):
        """Matplotlib figure'u QPixmap'e cevirir."""
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        q_img = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
        return QPixmap.fromImage(q_img)


# OpenCV import (change_view'de kullaniliyor)
import cv2