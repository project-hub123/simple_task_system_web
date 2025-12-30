"""
Автор: Федотова Анастасия Алексеевна
Тема ВКР:
Автоматическая генерация и проверка учебных заданий
по языку программирования Python с помощью нейронных сетей
(на примере ЧОУ ВО «Московский университет имени С.Ю. Витте»)
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QAction
)
from PyQt5.QtCore import Qt

from ml.task_generator import generate_task
from ml.predict import predict
from ml.database import save_result


class MainWindow(QMainWindow):
    def __init__(self, user: dict):
        super().__init__()

        self.user = user
        self.task = None

        self.setWindowTitle(
            f"Интеллектуальный сервис | {user['username']} ({user['role']})"
        )
        self.setMinimumSize(800, 600)

        self._create_menu()
        self._create_ui()

    # -------------------------------------------------
    # МЕНЮ
    # -------------------------------------------------

    def _create_menu(self):
        menu = self.menuBar()

        help_menu = menu.addMenu("Справка")

        about_action = QAction("О системе", self)
        about_action.triggered.connect(self.show_about)

        help_menu.addAction(about_action)

    # -------------------------------------------------
    # ОСНОВНОЙ ИНТЕРФЕЙС
    # -------------------------------------------------

    def _create_ui(self):
        central = QWidget()
        layout = QVBoxLayout()

        self.label_task = QLabel("Задание не сгенерировано")
        self.label_task.setWordWrap(True)
        self.label_task.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.text_solution = QTextEdit()
        self.text_solution.setPlaceholderText(
            "Введите решение на Python.\n"
            "Результат должен быть записан в переменную result."
        )

        btn_layout = QHBoxLayout()

        self.btn_generate = QPushButton("Сгенерировать задание")
        self.btn_generate.clicked.connect(self.generate_task)

        self.btn_check = QPushButton("Проверить решение")
        self.btn_check.clicked.connect(self.check_solution)

        btn_layout.addWidget(self.btn_generate)
        btn_layout.addWidget(self.btn_check)

        layout.addWidget(self.label_task)
        layout.addWidget(self.text_solution)
        layout.addLayout(btn_layout)

        central.setLayout(layout)
        self.setCentralWidget(central)

    # -------------------------------------------------
    # ЛОГИКА
    # -------------------------------------------------

    def generate_task(self):
        try:
            self.task = generate_task()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return

        self.label_task.setText(self.task["task_text"])
        self.text_solution.clear()

    def check_solution(self):
        if not self.task:
            QMessageBox.warning(self, "Ошибка", "Сначала сгенерируйте задание")
            return

        user_code = self.text_solution.toPlainText().strip()

        if not user_code:
            QMessageBox.warning(self, "Ошибка", "Введите решение")
            return

        result_text = predict(self.task, user_code)

        is_correct = result_text.startswith("✅")

        save_result(
            username=self.user["username"],
            task_text=self.task["task_text"],
            task_type=self.task["task_type"],
            user_code=user_code,
            is_correct=is_correct,
            feedback=result_text
        )

        QMessageBox.information(self, "Результат проверки", result_text)

    # -------------------------------------------------
    # СПРАВКА
    # -------------------------------------------------

    def show_about(self):
        QMessageBox.information(
            self,
            "О системе",
            "Интеллектуальный сервис автоматической генерации\n"
            "и проверки учебных заданий по Python.\n\n"
            "Автор: Федотова Анастасия Алексеевна\n"
            "ВКР, ЧОУ ВО «МУ им. С.Ю. Витте»"
        )
