from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QAction
)
from PyQt5.QtCore import Qt

from ml.task_generator import generate_task
from ml.predict import predict
from ml.database import save_result


class MainWindow(QMainWindow):
    def __init__(self, user: dict, on_logout):
        super().__init__()

        self.user = user
        self.on_logout = on_logout
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

        file_menu = menu.addMenu("Файл")

        logout_action = QAction("Сменить пользователя", self)
        logout_action.triggered.connect(self.logout)

        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(logout_action)
        file_menu.addAction(exit_action)

        help_menu = menu.addMenu("Справка")

        about_action = QAction("О системе", self)
        about_action.triggered.connect(self.show_about)

        help_menu.addAction(about_action)

    # -------------------------------------------------
    # UI
    # -------------------------------------------------

    def _create_ui(self):
        central = QWidget()
        layout = QVBoxLayout()

        # ТЕКСТ ЗАДАНИЯ + ИСХОДНЫЕ ДАННЫЕ
        self.label_task = QLabel("Задание не сгенерировано")
        self.label_task.setWordWrap(True)
        self.label_task.setAlignment(Qt.AlignTop)
        self.label_task.setStyleSheet(
            "font-size: 14px;"
            "font-weight: bold;"
            "padding: 8px;"
            "border: 1px solid #ccc;"
        )

        # ПОЛЕ ДЛЯ РЕШЕНИЯ
        self.text_solution = QTextEdit()
        self.text_solution.setPlaceholderText(
            "Введите решение на Python.\n\n"
            "❗ Не создавайте входные данные вручную.\n"
            "❗ Используйте переменные text или data.\n\n"
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
        except Exception:
            QMessageBox.warning(
                self,
                "ML недоступна",
                "Модель классификации не найдена.\n"
                "Используется резервная генерация заданий."
            )
            self.task = generate_task(use_ml=False)

        self._show_task()
        self.text_solution.clear()

    def _show_task(self):
        """
        Отображает задание ВМЕСТЕ с исходными данными
        """
        task_text = self.task["task_text"]
        task_type = self.task["task_type"]
        input_data = self.task.get("input_data", "")

        # Формируем отображение входных данных
        if task_type.startswith("text"):
            data_block = f'text = "{input_data}"'
        else:
            data_block = f"data = {input_data}"

        full_text = (
            f"{task_text}\n\n"
            f"Исходные данные:\n"
            f"{data_block}"
        )

        self.label_task.setText(full_text)

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
    # ВЫХОД
    # -------------------------------------------------

    def logout(self):
        self.close()
        self.on_logout()

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
            "ВКР, МУ им. С.Ю. Витте"
        )
