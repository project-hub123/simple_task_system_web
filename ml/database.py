"""
Автор: Федотова Анастасия Алексеевна
Тема ВКР:
Автоматическая генерация и проверка учебных заданий
по языку программирования Python
(на примере ЧОУ ВО «Московский университет имени С.Ю. Витте»)

Назначение:
Модуль работы с локальной БД SQLite.
Хранение пользователей, ролей, заданий и результатов.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "system.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


# -------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ БД
# -------------------------------------------------

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # USERS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # RESULTS
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

    # LOGS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------
# ПОЛЬЗОВАТЕЛИ
# -------------------------------------------------

def add_user(username: str, password_hash: str, role: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO users (username, password_hash, role)
        VALUES (?, ?, ?)
    """, (username, password_hash, role))
    conn.commit()
    conn.close()


def get_user(username: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT username, password_hash, role
        FROM users
        WHERE username = ?
    """, (username,))
    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "username": row[0],
            "password_hash": row[1],
            "role": row[2]
        }
    return None


def get_all_users():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT username, role
        FROM users
        ORDER BY username
    """)
    rows = cur.fetchall()
    conn.close()
    return [{"username": r[0], "role": r[1]} for r in rows]


# -------------------------------------------------
# РЕЗУЛЬТАТЫ
# -------------------------------------------------

def save_result(username, task_text, task_type, user_code, is_correct, feedback):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO results (
            username, task_text, task_type,
            user_code, is_correct, feedback, timestamp
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


def get_results_by_user(username):
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

    return [{
        "task_text": r[0],
        "task_type": r[1],
        "is_correct": bool(r[2]),
        "feedback": r[3],
        "timestamp": r[4]
    } for r in rows]


# -------------------------------------------------
# СТАТИСТИКА (ПРЕПОДАВАТЕЛЬ)
# -------------------------------------------------

def get_students_statistics():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT username,
               COUNT(*) AS attempts,
               SUM(is_correct) AS correct,
               MAX(timestamp) AS last_attempt
        FROM results
        GROUP BY username
    """)
    rows = cur.fetchall()
    conn.close()

    return [{
        "username": r[0],
        "attempts": r[1],
        "correct": r[2] or 0,
        "last_attempt": r[3]
    } for r in rows]


# -------------------------------------------------
# ЛОГИ (АДМИН)
# -------------------------------------------------

def add_log(message: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO logs (message, timestamp)
        VALUES (?, ?)
    """, (message, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def get_logs():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT message, timestamp
        FROM logs
        ORDER BY timestamp DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [{"message": r[0], "timestamp": r[1]} for r in rows]
