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
    Обеспечивает вход, выход и смену пользователя.
    """

    # Инициализация БД и пользователей
    init_system()

    app = QApplication(sys.argv)

    windows = {}

    # -------------------------------------------------
    # Показ окна входа
    # -------------------------------------------------
    def show_login():
        login_window = LoginWindow(on_login_success)
        login_window.show()
        windows["login"] = login_window

    # -------------------------------------------------
    # После успешного входа
    # -------------------------------------------------
    def on_login_success(user: dict):
        main_window = MainWindow(
            user=user,
            on_logout=show_login
        )
        main_window.show()
        windows["main"] = main_window

    # Запуск с окна входа
    show_login()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
