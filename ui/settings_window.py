"""
Автор: Федотова Анастасия Алексеевна
Тема ВКР:
Автоматическая генерация и проверка учебных заданий
по языку программирования Python
(на примере ЧОУ ВО «Московский университет имени С.Ю. Витте»)

Назначение:
Окно настроек приложения.
Используется администратором или преподавателем.
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QCheckBox, QSpinBox, QComboBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt


class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Настройки приложения")
        self.setFixedSize(420, 380)

        layout = QVBoxLayout()
        layout.setSpacing(12)

        # =====================================
        # ЗАГОЛОВОК
        # =====================================

        title = QLabel("Параметры системы")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 16px;"
            "font-weight: bold;"
        )
        layout.addWidget(title)

        # =====================================
        # 1. Использовать ML
        # =====================================

        self.chk_use_ml = QCheckBox("Использовать ML-классификацию заданий")
        self.chk_use_ml.setChecked(True)
        layout.addWidget(self.chk_use_ml)

        # =====================================
        # 2. Количество попыток
        # =====================================

        attempts_layout = QHBoxLayout()
        lbl_attempts = QLabel("Количество попыток на задание:")
        self.spin_attempts = QSpinBox()
        self.spin_attempts.setRange(1, 10)
        self.spin_attempts.setValue(3)

        attempts_layout.addWidget(lbl_attempts)
        attempts_layout.addWidget(self.spin_attempts)
        layout.addLayout(attempts_layout)

        # =====================================
        # 3. Уровень сложности
        # =====================================

        difficulty_layout = QHBoxLayout()
        lbl_difficulty = QLabel("Уровень сложности заданий:")
        self.combo_difficulty = QComboBox()
        self.combo_difficulty.addItems([
            "Базовый",
            "Средний",
            "Продвинутый"
        ])

        difficulty_layout.addWidget(lbl_difficulty)
        difficulty_layout.addWidget(self.combo_difficulty)
        layout.addLayout(difficulty_layout)

        # =====================================
        # 4. Логирование
        # =====================================

        self.chk_logging = QCheckBox("Вести журнал действий системы")
        self.chk_logging.setChecked(True)
        layout.addWidget(self.chk_logging)

        # =====================================
        # 5. Автосохранение результатов
        # =====================================

        self.chk_autosave = QCheckBox("Автоматически сохранять результаты")
        self.chk_autosave.setChecked(True)
        layout.addWidget(self.chk_autosave)

        # =====================================
        # КНОПКИ
        # =====================================

        btn_layout = QHBoxLayout()

        self.btn_save = QPushButton("Сохранить")
        self.btn_save.clicked.connect(self.save_settings)

        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_close)

        layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    # -------------------------------------------------
    # СОХРАНЕНИЕ (ПОКА ЛОГИЧЕСКОЕ)
    # -------------------------------------------------

    def save_settings(self):
        settings = {
            "use_ml": self.chk_use_ml.isChecked(),
            "max_attempts": self.spin_attempts.value(),
            "difficulty": self.combo_difficulty.currentText(),
            "logging": self.chk_logging.isChecked(),
            "autosave": self.chk_autosave.isChecked()
        }

        QMessageBox.information(
            self,
            "Настройки сохранены",
            "Параметры системы успешно сохранены.\n\n"
            f"{settings}"
        )
