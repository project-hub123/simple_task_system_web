from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QAction, QTabWidget
)
from PyQt5.QtCore import Qt

from ml.task_generator import generate_task
from ml.predict import predict
from ml.database import save_result

from ui.teacher_panel import TeacherPanel
from ui.admin_panel import AdminPanel
from ui.settings_window import SettingsWindow


class MainWindow(QMainWindow):
    def __init__(self, user: dict, on_logout):
        super().__init__()

        self.user = user
        self.on_logout = on_logout
        self.task = None
        self.settings_window = None

        self.setWindowTitle(
            f"Интеллектуальный сервис | {user['username']} ({user['role']})"
        )
        self.setMinimumSize(900, 650)

        self._create_menu()
        self._create_ui()

    # =================================================
    # МЕНЮ
    # =================================================

    def _create_menu(self):
        menu = self.menuBar()

        # ---------- Файл ----------
        file_menu = menu.addMenu("Файл")

        logout_action = QAction("Сменить пользователя", self)
        logout_action.triggered.connect(self.logout)

        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(logout_action)
        file_menu.addAction(exit_action)

        # ---------- Настройки ----------
        settings_menu = menu.addMenu("Настройки")

        settings_action = QAction("Параметры системы", self)
        settings_action.triggered.connect(self.open_settings)

        settings_menu.addAction(settings_action)

        # ---------- Справка ----------
        help_menu = menu.addMenu("Справка")

        about_action = QAction("О системе", self)
        about_action.triggered.connect(self.show_about)

        help_menu.addAction(about_action)

    # =================================================
    # UI
    # =================================================

    def _create_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()

        self.tabs = QTabWidget()

        # -----------------------------
        # ВКЛАДКА СТУДЕНТА
        # -----------------------------

        task_tab = QWidget()
        task_layout = QVBoxLayout()

        self.label_task = QLabel("Задание не сгенерировано")
        self.label_task.setWordWrap(True)
        self.label_task.setAlignment(Qt.AlignTop)
        self.label_task.setStyleSheet(
            "font-size: 14px;"
            "font-weight: bold;"
            "padding: 10px;"
            "border: 1px solid #bbb;"
            "background-color: #fafafa;"
        )

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

        task_layout.addWidget(self.label_task)
        task_layout.addWidget(self.text_solution)
        task_layout.addLayout(btn_layout)

        task_tab.setLayout(task_layout)
        self.tabs.addTab(task_tab, "Задания")

        # -----------------------------
        # ВКЛАДКА ПРЕПОДАВАТЕЛЯ
        # -----------------------------

        if self.user["role"] == "teacher":
            self.tabs.addTab(
                TeacherPanel(),
                "Статистика студентов"
            )

        # -----------------------------
        # ВКЛАДКА АДМИНИСТРАТОРА
        # -----------------------------

        if self.user["role"] == "admin":
            self.tabs.addTab(
                AdminPanel(),
                "Пользователи"
            )

        main_layout.addWidget(self.tabs)
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    # =================================================
    # ЛОГИКА
    # =================================================

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
        task_text = self.task["task_text"]
        task_type = self.task["task_type"]
        input_data = self.task.get("input_data", "")

        if task_type.startswith("text"):
            data_block = f'text = "{input_data}"'
        else:
            data_block = f"data = {input_data}"

        self.label_task.setText(
            f"{task_text}\n\n"
            f"Исходные данные:\n"
            f"{data_block}"
        )

    def check_solution(self):
        if not self.task:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Сначала сгенерируйте задание"
            )
            return

        user_code = self.text_solution.toPlainText().strip()

        if not user_code:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Введите решение"
            )
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

        QMessageBox.information(
            self,
            "Результат проверки",
            result_text
        )

    # =================================================
    # НАСТРОЙКИ
    # =================================================

    def open_settings(self):
        if self.settings_window is None:
            self.settings_window = SettingsWindow()
        self.settings_window.show()

    # =================================================
    # ВЫХОД
    # =================================================

    def logout(self):
        self.close()
        self.on_logout()

    # =================================================
    # СПРАВКА
    # =================================================

    def show_about(self):
        QMessageBox.information(
            self,
            "О системе",
            "Интеллектуальный сервис автоматической генерации и проверки "
            "учебных заданий по языку программирования Python.\n\n"
            "Система предназначена для использования в учебном процессе "
            "и обеспечивает автоматическую генерацию заданий, проверку "
            "корректности решений, хранение и анализ результатов, а также "
            "поддержку ролей пользователей.\n\n"
            "Автор: Федотова Анастасия Алексеевна\n"
            "Выпускная квалификационная работа\n"
            "ЧОУ ВО «Московский университет имени С.Ю. Витте»"
        )
