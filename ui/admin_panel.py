from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHBoxLayout, QLineEdit,
    QPushButton, QMessageBox, QComboBox,
    QFileDialog
)
from PyQt5.QtCore import Qt

import os
import shutil
import subprocess
import pandas as pd
from datetime import datetime

from ml.database import (
    get_all_users,
    add_user,
    update_user_password,
    set_user_active,
    log_admin_action,
    get_admin_logs
)

# -----------------------------
# Пути
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "train_data.csv")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_model.py")

REQUIRED_COLUMNS = {"task_text", "task_type"}

os.makedirs(DATA_DIR, exist_ok=True)


class AdminPanel(QWidget):
    def __init__(self, admin_username="admin"):
        super().__init__()

        self.admin_username = admin_username

        self.setWindowTitle("Панель администратора")
        self.resize(900, 600)

        main_layout = QVBoxLayout()

        title = QLabel("Панель администратора системы")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(title)

        # -----------------------------
        # Таблица пользователей
        # -----------------------------
        self.users_table = QTableWidget()
        self.users_table.setColumnCount(3)
        self.users_table.setHorizontalHeaderLabels([
            "Пользователь", "Роль", "Статус"
        ])
        self.users_table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.users_table)

        # -----------------------------
        # Форма добавления пользователя
        # -----------------------------
        add_layout = QHBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Имя пользователя")
        add_layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Пароль")
        self.password_input.setEchoMode(QLineEdit.Password)
        add_layout.addWidget(self.password_input)

        self.role_input = QComboBox()
        self.role_input.addItems(["student", "teacher", "admin"])
        add_layout.addWidget(self.role_input)

        self.btn_add_user = QPushButton("Добавить пользователя")
        self.btn_add_user.clicked.connect(self.add_user)
        add_layout.addWidget(self.btn_add_user)

        main_layout.addLayout(add_layout)

        # -----------------------------
        # Управление пользователем
        # -----------------------------
        manage_layout = QHBoxLayout()

        self.btn_reset_password = QPushButton("Сбросить пароль")
        self.btn_reset_password.clicked.connect(self.reset_password)
        manage_layout.addWidget(self.btn_reset_password)

        self.btn_toggle_active = QPushButton("Блок / Разблок")
        self.btn_toggle_active.clicked.connect(self.toggle_active)
        manage_layout.addWidget(self.btn_toggle_active)

        main_layout.addLayout(manage_layout)

        # -----------------------------
        # Работа с датасетом и моделью
        # -----------------------------
        ml_layout = QHBoxLayout()

        self.btn_upload_dataset = QPushButton("Загрузить датасет")
        self.btn_upload_dataset.clicked.connect(self.upload_dataset)
        ml_layout.addWidget(self.btn_upload_dataset)

        self.btn_train_model = QPushButton("Переобучить модель")
        self.btn_train_model.clicked.connect(self.train_model)
        ml_layout.addWidget(self.btn_train_model)

        main_layout.addLayout(ml_layout)

        # -----------------------------
        # Журнал действий
        # -----------------------------
        log_title = QLabel("Журнал действий администратора")
        log_title.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(log_title)

        self.log_table = QTableWidget()
        self.log_table.setColumnCount(3)
        self.log_table.setHorizontalHeaderLabels([
            "Дата", "Администратор", "Действие"
        ])
        self.log_table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.log_table)

        self.setLayout(main_layout)

        self.load_users()
        self.load_logs()

    # =================================================
    # Пользователи
    # =================================================

    def load_users(self):
        users = get_all_users()
        self.users_table.setRowCount(len(users))

        for row, u in enumerate(users):
            self.users_table.setItem(row, 0, QTableWidgetItem(u["username"]))
            self.users_table.setItem(row, 1, QTableWidgetItem(u["role"]))
            status = "Активен" if u["is_active"] else "Заблокирован"
            self.users_table.setItem(row, 2, QTableWidgetItem(status))

    def add_user(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        role = self.role_input.currentText()

        if not username or not password:
            QMessageBox.warning(self, "Ошибка", "Введите имя и пароль")
            return

        try:
            add_user(username, password, role)
            log_admin_action(
                self.admin_username,
                f"Добавлен пользователь {username}"
            )
            self.username_input.clear()
            self.password_input.clear()
            self.load_users()
            self.load_logs()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def get_selected_user(self):
        row = self.users_table.currentRow()
        if row == -1:
            return None
        return self.users_table.item(row, 0).text()

    def reset_password(self):
        username = self.get_selected_user()
        if not username:
            QMessageBox.warning(self, "Ошибка", "Выберите пользователя")
            return

        new_password, ok = QFileDialog.getText(
            self, "Сброс пароля", "Введите новый пароль:"
        )
        if not ok or not new_password:
            return

        update_user_password(username, new_password)
        log_admin_action(
            self.admin_username,
            f"Сброшен пароль пользователя {username}"
        )
        self.load_logs()
        QMessageBox.information(self, "Готово", "Пароль обновлён")

    def toggle_active(self):
        username = self.get_selected_user()
        if not username:
            QMessageBox.warning(self, "Ошибка", "Выберите пользователя")
            return

        set_user_active(username)
        log_admin_action(
            self.admin_username,
            f"Изменён статус пользователя {username}"
        )
        self.load_users()
        self.load_logs()

    # =================================================
    # Датасет и модель
    # =================================================

    def upload_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбор CSV", "", "CSV (*.csv)"
        )
        if not file_path:
            return

        df = pd.read_csv(file_path)
        if not REQUIRED_COLUMNS.issubset(df.columns):
            QMessageBox.critical(
                self,
                "Ошибка",
                "Некорректная структура датасета"
            )
            return

        shutil.copy(file_path, DATASET_PATH)
        log_admin_action(
            self.admin_username,
            "Загружен обучающий датасет"
        )
        self.load_logs()
        QMessageBox.information(self, "Готово", "Датасет загружен")

    def train_model(self):
        try:
            subprocess.run(
                ["python", TRAIN_SCRIPT],
                check=True
            )
            log_admin_action(
                self.admin_username,
                "Выполнено переобучение модели"
            )
            self.load_logs()
            QMessageBox.information(
                self, "Готово", "Модель успешно переобучена"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    # =================================================
    # Логи
    # =================================================

    def load_logs(self):
        logs = get_admin_logs()
        self.log_table.setRowCount(len(logs))

        for row, log in enumerate(logs):
            self.log_table.setItem(row, 0, QTableWidgetItem(log["timestamp"]))
            self.log_table.setItem(row, 1, QTableWidgetItem(log["admin"]))
            self.log_table.setItem(row, 2, QTableWidgetItem(log["action"]))
