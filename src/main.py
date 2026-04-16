import sys
from PyQt5.QtWidgets import QApplication
from ui_manager import UIManager

app = QApplication(sys.argv)
window = UIManager()
window.show()
sys.exit(app.exec_())
