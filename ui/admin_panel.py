from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHBoxLayout, QLineEdit,
    QPushButton, QMessageBox, QComboBox
)
from ml.database import get_all_users, add_user_simple


class AdminPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Администрирование пользователей")
        self.resize(600, 400)

        main_layout = QVBoxLayout()

        title = QLabel("Администрирование пользователей")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(title)

        # ---------- Таблица ----------
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels([
            "Имя пользователя", "Роль"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.table)

        # ---------- Форма добавления ----------
        form_layout = QHBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Имя пользователя")
        form_layout.addWidget(self.username_input)

        self.role_input = QComboBox()
        self.role_input.addItems(["user", "student", "teacher", "admin"])
        form_layout.addWidget(self.role_input)

        self.add_button = QPushButton("Добавить")
        self.add_button.clicked.connect(self.add_user)
        form_layout.addWidget(self.add_button)

        main_layout.addLayout(form_layout)

        self.setLayout(main_layout)

        self.load_users()

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
            # 🔥 ВАЖНО: используем ТОЛЬКО add_user_simple
            add_user_simple(username, role)
            self.username_input.clear()
            self.load_users()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
