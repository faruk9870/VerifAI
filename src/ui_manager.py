import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
    QFileDialog, QButtonGroup, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt
from PIL import Image

class UIManager(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("verifAI")
        self.setGeometry(100, 100, 480, 650) 

        # --- MODERN KARANLIK TEMA VE TÜM EFEKTLER (QSS) ---
        self.setStyleSheet("""
            QWidget {
                background-color: #11111b; 
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel#TitleLabel {
                font-size: 22px;
                font-weight: 800;
                color: #cba6f7; 
                margin-bottom: 5px;
            }
            QLabel#SubtitleLabel {
                font-size: 12px;
                color: #a6adc8;
                margin-bottom: 15px;
            }
            QLabel#ImageLabel {
                background-color: #1e1e2e;
                border: 2px dashed #45475a;
                border-radius: 10px;
                font-size: 14px;
                color: #7f849c;
            }
            QLabel#InfoLabel {
                background-color: #1e1e2e;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
            }
            
            /* --- ANA AKSİYON BUTONU EFEKTLERİ --- */
            QPushButton#ActionBtn {
                background-color: #cba6f7;
                color: #11111b;
                font-weight: bold;
                font-size: 15px;
                padding: 14px;
                border-radius: 8px;
                border: none;
            }
            QPushButton#ActionBtn:hover {
                background-color: #b4befe; /* Üzerine gelince parlar */
            }
            QPushButton#ActionBtn:pressed {
                background-color: #89b4fa; /* Tıklayınca renk değiştirir */
            }

            /* --- FORMAT SEÇİCİ (TOGGLE) BUTON EFEKTLERİ --- */
            QPushButton#ToggleBtn {
                background-color: #1e1e2e;
                color: #a6adc8;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                border: 2px solid transparent;
            }
            QPushButton#ToggleBtn:checked {
                background-color: #89b4fa; /* Aktif (Seçili) durum efekti */
                color: #11111b;
            }
            QPushButton#ToggleBtn:hover:!checked {
                background-color: #313244; /* Sadece seçili değilken üzerine gelme efekti */
            }
        """)

        # --- YAPI VE YERLEŞİM (LAYOUT) ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(15)

        # Başlık ve Alt Başlık
        self.title = QLabel("AI Görsel Denetimi")
        self.title.setObjectName("TitleLabel")
        self.title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title)

        self.subtitle = QLabel("Analiz edilecek dosya türünü seçin ve yükleyin")
        self.subtitle.setObjectName("SubtitleLabel")
        self.subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.subtitle)

        # --- FORMAT SEÇİCİ (Hap Tasarımlı Butonlar) ---
        format_layout = QHBoxLayout()
        
        self.radio_png = QPushButton("PNG")
        self.radio_png.setCursor(Qt.PointingHandCursor)
        self.radio_png.setCheckable(True) # Basılı kalabilme özelliği
        self.radio_png.setChecked(True)   # Varsayılan olarak PNG seçili
        self.radio_png.setObjectName("ToggleBtn")
        
        self.radio_jpg = QPushButton("JPEG")
        self.radio_jpg.setCursor(Qt.PointingHandCursor)
        self.radio_jpg.setCheckable(True) # Basılı kalabilme özelliği
        self.radio_jpg.setObjectName("ToggleBtn")

        # Butonları grupluyoruz ki sadece biri basılı kalabilsin
        self.format_group = QButtonGroup()
        self.format_group.addButton(self.radio_png)
        self.format_group.addButton(self.radio_jpg)

        format_layout.addWidget(self.radio_png)
        format_layout.addWidget(self.radio_jpg)
        main_layout.addLayout(format_layout)

        # --- FOTOĞRAF ALANI ---
        self.label = QLabel("Sürükle bırak veya klasörden seç")
        self.label.setObjectName("ImageLabel")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(420, 300)
        self.add_shadow(self.label)
        main_layout.addWidget(self.label, alignment=Qt.AlignCenter)

        # --- BİLGİ ALANI ---
        self.info = QLabel("Görüntü analiz sonuçları bekleniyor...")
        self.info.setObjectName("InfoLabel")
        self.info.setAlignment(Qt.AlignCenter)
        self.add_shadow(self.info)
        main_layout.addWidget(self.info)

        # --- YÜKLE BUTONU ---
        self.button = QPushButton("📸 Fotoğraf Yükle ve Analiz Et")
        self.button.setObjectName("ActionBtn")
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.clicked.connect(self.load_image)
        main_layout.addWidget(self.button)

        self.setLayout(main_layout)

    def add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80)) 
        shadow.setOffset(0, 4)
        widget.setGraphicsEffect(shadow)

    def load_image(self):
        # Seçili butona göre filtre ayarı
        if self.radio_png.isChecked():
            file_filter = "PNG Images (*.png)"
            expected_format = "PNG"
        else:
            file_filter = "JPEG Images (*.jpg *.jpeg)"
            expected_format = "JPEG"

        file_path, _ = QFileDialog.getOpenFileName(self, "Analiz İçin Dosya Seç", "", file_filter)

        if file_path:
            # Resmi Göster
            pixmap = QPixmap(file_path)
            self.label.setPixmap(pixmap.scaled(400, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # Analiz Et
            try:
                img = Image.open(file_path)
                width, height = img.size
                actual_format = img.format

                if (width, height) == (224, 224):
                    size_status = "<span style='color: #a6e3a1;'>✅ Uygun (224x224)</span>"
                else:
                    size_status = f"<span style='color: #f38ba8;'>❌ Hatalı Boyut ({width}x{height})</span>"

                if actual_format == expected_format:
                    format_status = f"<span style='color: #a6e3a1;'>✅ Doğrulandı ({actual_format})</span>"
                else:
                    format_status = f"<span style='color: #f38ba8;'>❌ UYUŞMAZLIK! (Seçilen: {expected_format}, Gerçek: {actual_format})</span>"

                self.info.setText(
                    f"<div style='line-height: 1.5;'>"
                    f"<b>Dosya Formatı Tespiti:</b> {format_status}<br>"
                    f"<b>Görüntü Çözünürlüğü:</b> {size_status}"
                    f"</div>"
                )
            except Exception as e:
                self.info.setText(f"<span style='color: #f38ba8;'>❌ Hata: Dosya okunamadı!<br>{str(e)}</span>")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UIManager()
    window.show()
    sys.exit(app.exec_())