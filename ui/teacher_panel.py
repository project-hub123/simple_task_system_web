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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "train_data.csv")

REQUIRED_COLUMNS = {"task_text", "task_type"}

os.makedirs(DATA_DIR, exist_ok=True)


class TeacherPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Панель преподавателя")
        self.setMinimumSize(750, 550)

        layout = QVBoxLayout()

        title = QLabel("Статистика студентов и управление данными")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        # ---------- Таблица ----------
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

        self.btn_upload_dataset = QPushButton("Загрузить обучающий датасет")
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
    # Статистика студентов
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
    # Экспорт
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
    # Датасет
    # -------------------------------------------------

    def update_dataset_status(self):
        if os.path.exists(DATASET_PATH):
            self.dataset_label.setText(
                f"Используется обучающий датасет: {os.path.basename(DATASET_PATH)}"
            )
        else:
            self.dataset_label.setText("Обучающий датасет не загружен")

    def upload_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбор CSV-файла с обучающим датасетом",
            "",
            "CSV файлы (*.csv)"
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            missing = REQUIRED_COLUMNS - set(df.columns)

            if missing:
                raise ValueError(
                    "Некорректная структура датасета. "
                    f"Отсутствуют колонки: {', '.join(missing)}"
                )

            shutil.copy(file_path, DATASET_PATH)

            QMessageBox.information(
                self,
                "Успешно",
                "Датасет успешно загружен и будет использован при следующем обучении модели."
            )

            self.update_dataset_status()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки датасета", str(e))
