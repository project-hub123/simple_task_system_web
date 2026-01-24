from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHBoxLayout
)
from PyQt5.QtCore import Qt

import os
import shutil
import pandas as pd

from ml.database import get_students_statistics

# ======================================================
# ПУТИ
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "train_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ======================================================
# ПАНЕЛЬ ПРЕПОДАВАТЕЛЯ
# ======================================================

class TeacherPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Панель преподавателя")
        self.setMinimumSize(750, 550)

        layout = QVBoxLayout()

        title = QLabel("Статистика студентов и управление обучением")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        # ---------- Таблица статистики ----------
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Студент",
            "Попыток",
            "Верных решений",
            "Последняя попытка"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)

        # ---------- Кнопки ----------
        buttons_layout = QHBoxLayout()

        self.btn_export = QPushButton("Экспорт статистики в Excel")
        self.btn_export.clicked.connect(self.export_to_excel)
        buttons_layout.addWidget(self.btn_export)

        self.btn_upload_dataset = QPushButton("Загрузить датасет для обучения модели")
        self.btn_upload_dataset.clicked.connect(self.upload_dataset)
        buttons_layout.addWidget(self.btn_upload_dataset)

        self.dataset_label = QLabel("")
        self.dataset_label.setAlignment(Qt.AlignLeft)

        layout.addWidget(title)
        layout.addWidget(self.table)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.dataset_label)

        self.setLayout(layout)

        self.load_data()
        self.update_dataset_status()

    # -------------------------------------------------
    # ЗАГРУЗКА СТАТИСТИКИ
    # -------------------------------------------------

    def load_data(self):
        stats = get_students_statistics()
        self.table.setRowCount(len(stats))

        for row, s in enumerate(stats):
            self.table.setItem(row, 0, QTableWidgetItem(s["username"]))
            self.table.setItem(row, 1, QTableWidgetItem(str(s["attempts"])))
            self.table.setItem(row, 2, QTableWidgetItem(str(s["correct"])))
            self.table.setItem(row, 3, QTableWidgetItem(str(s["last_attempt"])))

    # -------------------------------------------------
    # ЭКСПОРТ СТАТИСТИКИ
    # -------------------------------------------------

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
            QMessageBox.information(self, "Готово", "Файл успешно сохранён")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    # -------------------------------------------------
    # ДАТАСЕТ ДЛЯ ОБУЧЕНИЯ (ОПЦИОНАЛЬНО)
    # -------------------------------------------------

    def update_dataset_status(self):
        if os.path.exists(DATASET_PATH):
            self.dataset_label.setText(
                f"Загружен датасет для обучения модели: {os.path.basename(DATASET_PATH)}"
            )
        else:
            self.dataset_label.setText(
                "Датасет для обучения модели не загружен (не обязателен для работы системы)"
            )

    def upload_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбор CSV-файла для обучения модели",
            "",
            "CSV файлы (*.csv)"
        )

        if not file_path:
            return

        try:
            # Проверяем, что это CSV, без жёсткой привязки к структуре
            pd.read_csv(file_path)

            shutil.copy(file_path, DATASET_PATH)

            QMessageBox.information(
                self,
                "Успешно",
                "Датасет загружен и может быть использован при обучении модели."
            )

            self.update_dataset_status()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки датасета", str(e))
