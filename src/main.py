import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("VerifAI")
window.resize(400, 200)
layout = QVBoxLayout(window)
label = QLabel("VerifAI - AI Gorsel Tespit Sistemi")
label.setAlignment(Qt.AlignCenter)
layout.addWidget(label)
window.show()
sys.exit(app.exec_())
