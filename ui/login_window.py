"""
Автор: Федотова Анастасия Алексеевна
Тема ВКР:
Автоматическая генерация и проверка учебных заданий
по языку программирования Python с помощью нейронных сетей
(на примере ЧОУ ВО «Московский университет имени С.Ю. Витте»)

Назначение:
Окно входа пользователя в десктопное приложение.
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt

from ml.auth import login, init_system


class LoginWindow(QWidget):
    def __init__(self, on_login_success):
        super().__init__()

        self.on_login_success = on_login_success

        self.setWindowTitle("Вход в систему")
        self.setFixedSize(360, 220)

        # Инициализация БД и пользователей
        init_system()

        # ---------- UI ----------
        self.label_title = QLabel("Интеллектуальный сервис\nпроверки заданий Python")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.label_user = QLabel("Имя пользователя:")
        self.input_user = QLineEdit()
        self.input_user.setPlaceholderText("student / teacher / admin")

        self.btn_login = QPushButton("Войти")
        self.btn_login.clicked.connect(self.handle_login)

        self.btn_exit = QPushButton("Выход")
        self.btn_exit.clicked.connect(self.close)

        # ---------- Layout ----------
        layout = QVBoxLayout()
        layout.addWidget(self.label_title)
        layout.addSpacing(10)
        layout.addWidget(self.label_user)
        layout.addWidget(self.input_user)
        layout.addSpacing(10)
        layout.addWidget(self.btn_login)
        layout.addWidget(self.btn_exit)

        self.setLayout(layout)

    # -------------------------------------------------
    # ЛОГИКА ВХОДА
    # -------------------------------------------------

    def handle_login(self):
        username = self.input_user.text().strip()

        if not username:
            QMessageBox.warning(self, "Ошибка", "Введите имя пользователя")
            return

        user = login(username)

        if not user:
            QMessageBox.critical(
                self,
                "Ошибка входа",
                "Пользователь не найден.\n"
                "Доступные пользователи:\n"
                "student, teacher, admin"
            )
            return

        QMessageBox.information(
            self,
            "Успешный вход",
            f"Вы вошли как:\n{user['username']} ({user['role']})"
        )

        self.on_login_success(user)
        self.close()
