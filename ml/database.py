"""
Автор: Федотова Анастасия Алексеевна
Тема ВКР:
Автоматическая генерация и проверка учебных заданий
по языку программирования Python с помощью нейронных сетей
(на примере ЧОУ ВО «Московский университет имени С.Ю. Витте»)

Назначение:
Модуль работы с локальной базой данных SQLite.
Обеспечивает хранение пользователей и результатов проверки заданий.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

# -------------------------------------------------
# ПУТЬ К БАЗЕ ДАННЫХ
# -------------------------------------------------

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "system.db"


# -------------------------------------------------
# ПОДКЛЮЧЕНИЕ К БД
# -------------------------------------------------

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


# -------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ БД
# -------------------------------------------------

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Таблица пользователей
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # Таблица результатов
    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            task_text TEXT NOT NULL,
            task_type TEXT NOT NULL,
            user_code TEXT NOT NULL,
            is_correct INTEGER NOT NULL,
            feedback TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------
# ПОЛЬЗОВАТЕЛИ
# -------------------------------------------------

def add_user(username: str, role: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR IGNORE INTO users (username, role)
        VALUES (?, ?)
    """, (username, role))

    conn.commit()
    conn.close()


def get_user(username: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT username, role FROM users
        WHERE username = ?
    """, (username,))

    row = cur.fetchone()
    conn.close()

    if row:
        return {"username": row[0], "role": row[1]}
    return None


def get_all_users():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT username, role FROM users
        ORDER BY username
    """)

    rows = cur.fetchall()
    conn.close()

    return [{"username": r[0], "role": r[1]} for r in rows]


# -------------------------------------------------
# РЕЗУЛЬТАТЫ ПРОВЕРКИ
# -------------------------------------------------

def save_result(
    username: str,
    task_text: str,
    task_type: str,
    user_code: str,
    is_correct: bool,
    feedback: str
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO results (
            username,
            task_text,
            task_type,
            user_code,
            is_correct,
            feedback,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        task_text,
        task_type,
        user_code,
        int(is_correct),
        feedback,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_results_by_user(username: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT task_text, task_type, is_correct, feedback, timestamp
        FROM results
        WHERE username = ?
        ORDER BY timestamp DESC
    """, (username,))

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "task_text": r[0],
            "task_type": r[1],
            "is_correct": bool(r[2]),
            "feedback": r[3],
            "timestamp": r[4],
        }
        for r in rows
    ]


def get_all_results():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT username, task_text, task_type, is_correct, timestamp
        FROM results
        ORDER BY timestamp DESC
    """)

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "username": r[0],
            "task_text": r[1],
            "task_type": r[2],
            "is_correct": bool(r[3]),
            "timestamp": r[4],
        }
        for r in rows
    ]
