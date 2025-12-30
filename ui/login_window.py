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
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt

from ml.auth import login, init_system


class LoginWindow(QWidget):
    def __init__(self, on_login_success):
        super().__init__()

        self.on_login_success = on_login_success

        self.setWindowTitle("Вход в систему")
        self.setFixedSize(420, 300)

        # Инициализация БД и пользователей
        init_system()

        # =============================
        # ЗАГОЛОВОК
        # =============================

        self.label_title = QLabel("Интеллектуальный сервис")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet(
            "font-size: 18px;"
            "font-weight: bold;"
        )

        self.label_subtitle = QLabel(
            "Автоматическая генерация и проверка\n"
            "учебных заданий по Python"
        )
        self.label_subtitle.setAlignment(Qt.AlignCenter)
        self.label_subtitle.setStyleSheet(
            "font-size: 12px;"
            "color: #555;"
        )

        # =============================
        # РАЗДЕЛИТЕЛЬ
        # =============================

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        # =============================
        # ПОЛЕ ВВОДА
        # =============================

        self.label_user = QLabel("Имя пользователя")
        self.label_user.setStyleSheet("font-weight: bold;")

        self.input_user = QLineEdit()
        self.input_user.setPlaceholderText("student / teacher / admin")
        self.input_user.setFixedHeight(32)

        hint = QLabel(
            "Пример:\n"
            "student — студент\n"
            "teacher — преподаватель\n"
            "admin — администратор"
        )
        hint.setStyleSheet(
            "font-size: 11px;"
            "color: #666;"
        )

        # =============================
        # КНОПКИ
        # =============================

        self.btn_login = QPushButton("Войти")
        self.btn_login.setFixedHeight(34)
        self.btn_login.setStyleSheet(
            "font-weight: bold;"
        )
        self.btn_login.clicked.connect(self.handle_login)

        self.btn_exit = QPushButton("Выход")
        self.btn_exit.setFixedHeight(30)
        self.btn_exit.clicked.connect(self.close)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_login)
        btn_layout.addWidget(self.btn_exit)

        # =============================
        # ОСНОВНОЙ LAYOUT
        # =============================

        layout = QVBoxLayout()
        layout.setSpacing(10)

        layout.addWidget(self.label_title)
        layout.addWidget(self.label_subtitle)
        layout.addSpacing(5)
        layout.addWidget(line)
        layout.addSpacing(10)

        layout.addWidget(self.label_user)
        layout.addWidget(self.input_user)
        layout.addWidget(hint)

        layout.addSpacing(15)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    # -------------------------------------------------
    # ЛОГИКА ВХОДА
    # -------------------------------------------------

    def handle_login(self):
        username = self.input_user.text().strip()

        if not username:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Введите имя пользователя"
            )
            return

        user = login(username)

        if not user:
            QMessageBox.critical(
                self,
                "Ошибка входа",
                "Пользователь не найден.\n\n"
                "Доступные пользователи:\n"
                "• student\n"
                "• teacher\n"
                "• admin"
            )
            return

        QMessageBox.information(
            self,
            "Успешный вход",
            f"Вы вошли как:\n{user['username']} ({user['role']})"
        )

        self.on_login_success(user)
        self.close()
