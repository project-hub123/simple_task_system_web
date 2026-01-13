from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt

from ml.auth import login, init_system


class LoginWindow(QWidget):
    def __init__(self, on_login_success):
        super().__init__()

        self.on_login_success = on_login_success

        self.setWindowTitle("Вход в систему")
        self.setFixedSize(420, 340)

        # Инициализация БД
        init_system()

        self.label_title = QLabel("Интеллектуальный сервис")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.label_subtitle = QLabel(
            "Автоматическая генерация и проверка\n"
            "учебных заданий по Python"
        )
        self.label_subtitle.setAlignment(Qt.AlignCenter)
        self.label_subtitle.setStyleSheet("font-size: 12px; color: #555;")

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        self.label_user = QLabel("Имя пользователя")
        self.input_user = QLineEdit()
        self.input_user.setPlaceholderText("student / teacher / admin")

        self.label_pass = QLabel("Пароль")
        self.input_pass = QLineEdit()
        self.input_pass.setEchoMode(QLineEdit.Password)
        self.input_pass.setPlaceholderText("Введите пароль")

        self.btn_login = QPushButton("Войти")
        self.btn_login.clicked.connect(self.handle_login)

        self.btn_exit = QPushButton("Выход")
        self.btn_exit.clicked.connect(self.close)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_login)
        btns.addWidget(self.btn_exit)

        layout = QVBoxLayout()
        layout.addWidget(self.label_title)
        layout.addWidget(self.label_subtitle)
        layout.addWidget(line)
        layout.addWidget(self.label_user)
        layout.addWidget(self.input_user)
        layout.addWidget(self.label_pass)
        layout.addWidget(self.input_pass)
        layout.addSpacing(10)
        layout.addLayout(btns)

        self.setLayout(layout)

    # -------------------------------------------------

    def handle_login(self):
        username = self.input_user.text().strip()
        password = self.input_pass.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "Ошибка", "Введите логин и пароль")
            return

        user = login(username, password)

        if not user:
            QMessageBox.critical(self, "Ошибка входа", "Неверный логин или пароль")
            return

        QMessageBox.information(
            self,
            "Успешный вход",
            f"Вы вошли как {user['username']} ({user['role']})"
        )

        self.on_login_success(user)
        self.close()
