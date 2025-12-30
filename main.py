# main.py
# Автор: Федотова Анастасия Алексеевна
# Тема ВКР:
# «Автоматическая генерация и проверка учебных заданий
# по языку программирования Python с помощью нейронных сетей
# (на примере ЧОУ ВО „Московский университет имени С.Ю. Витте“)»

import sys
from PyQt5.QtWidgets import QApplication

from ui.login_window import LoginWindow
from ui.main_window import MainWindow
from ml.auth import init_system


def main():
    """
    Точка входа десктопного приложения.
    """

    # Инициализация БД и системных пользователей
    init_system()

    app = QApplication(sys.argv)

    main_window_ref = {"window": None}

    def on_login_success(user: dict):
        """
        Открытие главного окна после входа пользователя
        """
        window = MainWindow(user)
        window.show()
        main_window_ref["window"] = window

    # Окно входа
    login_window = LoginWindow(on_login_success)
    login_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
