from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt

import pandas as pd

from ml.database import get_students_statistics


class TeacherPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Панель преподавателя")
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout()

        title = QLabel("Статистика студентов")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Студент",
            "Попыток",
            "Верных решений",
            "Последняя попытка"
        ])

        self.btn_export = QPushButton("Экспорт в Excel (.xlsx)")
        self.btn_export.clicked.connect(self.export_to_excel)

        layout.addWidget(title)
        layout.addWidget(self.table)
        layout.addWidget(self.btn_export)

        self.setLayout(layout)

        self.load_data()

    # -----------------------------------------

    def load_data(self):
        stats = get_students_statistics()
        self.table.setRowCount(len(stats))

        for row, s in enumerate(stats):
            self.table.setItem(row, 0, QTableWidgetItem(s["username"]))
            self.table.setItem(row, 1, QTableWidgetItem(str(s["attempts"])))
            self.table.setItem(row, 2, QTableWidgetItem(str(s["correct"])))
            self.table.setItem(row, 3, QTableWidgetItem(str(s["last_attempt"])))

    # -----------------------------------------

    def export_to_excel(self):
        stats = get_students_statistics()

        if not stats:
            QMessageBox.warning(self, "Ошибка", "Нет данных для экспорта")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить файл",
            "students_statistics.xlsx",
            "Excel Files (*.xlsx)"
        )

        if not path:
            return

        df = pd.DataFrame(stats)
        df.columns = [
            "Студент",
            "Количество попыток",
            "Верных решений",
            "Последняя попытка"
        ]

        try:
            df.to_excel(path, index=False)
            QMessageBox.information(
                self,
                "Готово",
                "Файл успешно сохранён"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось сохранить файл:\n{e}"
            )
