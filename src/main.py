import sys
<<<<<<< HEAD
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
=======
from PyQt5.QtWidgets import QApplication
from ui_manager import UIManager

app = QApplication(sys.argv)

window = UIManager()
window.show()

sys.exit(app.exec_())
>>>>>>> e9ed43ce282d2329e2b39f7cd330ab5eaf36a57e
