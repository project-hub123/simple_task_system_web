from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHBoxLayout, QLineEdit,
    QPushButton, QMessageBox, QComboBox,
    QFileDialog
)
import os
import shutil
import pandas as pd

from ml.database import get_all_users, add_user_simple


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "train_data.csv")

REQUIRED_COLUMNS = {"task_text", "task_type"}

os.makedirs(DATA_DIR, exist_ok=True)


class AdminPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Администрирование")
        self.resize(700, 500)

        main_layout = QVBoxLayout()

        title = QLabel("Администрирование пользователей и данных")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(title)

        # ---------- Таблица пользователей ----------
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels([
            "Имя пользователя", "Роль"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.table)

        # ---------- Форма добавления пользователя ----------
        form_layout = QHBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Имя пользователя")
        form_layout.addWidget(self.username_input)

        self.role_input = QComboBox()
        self.role_input.addItems(["student", "teacher", "admin"])
        form_layout.addWidget(self.role_input)

        self.add_button = QPushButton("Добавить пользователя")
        self.add_button.clicked.connect(self.add_user)
        form_layout.addWidget(self.add_button)

        main_layout.addLayout(form_layout)

        # ---------- Загрузка датасета ----------
        dataset_layout = QHBoxLayout()

        self.dataset_label = QLabel("Обучающий датасет не загружен")
        dataset_layout.addWidget(self.dataset_label)

        self.upload_button = QPushButton("Загрузить датасет")
        self.upload_button.clicked.connect(self.upload_dataset)
        dataset_layout.addWidget(self.upload_button)

        main_layout.addLayout(dataset_layout)

        self.setLayout(main_layout)

        self.load_users()
        self.update_dataset_status()

    # -------------------------------------------------
    # Пользователи
    # -------------------------------------------------

    def load_users(self):
        users = get_all_users()
        self.table.setRowCount(len(users))

        for row, user in enumerate(users):
            self.table.setItem(row, 0, QTableWidgetItem(user["username"]))
            self.table.setItem(row, 1, QTableWidgetItem(user["role"]))

    def add_user(self):
        username = self.username_input.text().strip()
        role = self.role_input.currentText()

        if not username:
            QMessageBox.warning(self, "Ошибка", "Введите имя пользователя")
            return

        try:
            add_user_simple(username, role)
            self.username_input.clear()
            self.load_users()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    # -------------------------------------------------
    # Датасет
    # -------------------------------------------------

    def update_dataset_status(self):
        if os.path.exists(DATASET_PATH):
            self.dataset_label.setText(
                f"Используется датасет: {os.path.basename(DATASET_PATH)}"
            )
        else:
            self.dataset_label.setText("Обучающий датасет не загружен")

    def upload_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбор CSV-файла с датасетом",
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
                "Датасет успешно загружен и будет использован при обучении модели."
            )

            self.update_dataset_status()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки датасета", str(e))
