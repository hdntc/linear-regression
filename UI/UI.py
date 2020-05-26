import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi

class MainWindow(QMainWindow):
    def __init__(self, *args):
        super().__init__(*args)

        loadUi("ui.ui", self)
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())
