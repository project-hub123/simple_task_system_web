import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

# -------------------------------------------------
# ПУТЬ К БАЗЕ ДАННЫХ
# -------------------------------------------------

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "system.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# -------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ БД
# -------------------------------------------------

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # пользователи
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            is_active INTEGER DEFAULT 1
        )
    """)

    # результаты выполнения заданий
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

    # журнал действий администратора
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------
# ПОЛЬЗОВАТЕЛИ
# -------------------------------------------------

def add_user(username: str, password: str, role: str):
    password_hash = hash_password(password)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO users (username, password_hash, role)
        VALUES (?, ?, ?)
    """, (username, password_hash, role))

    conn.commit()
    conn.close()


def update_user_password(username: str, new_password: str):
    password_hash = hash_password(new_password)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE users
        SET password_hash = ?
        WHERE username = ?
    """, (password_hash, username))

    conn.commit()
    conn.close()


def set_user_active(username: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT is_active FROM users WHERE username = ?
    """, (username,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return

    new_status = 0 if row[0] else 1

    cur.execute("""
        UPDATE users
        SET is_active = ?
        WHERE username = ?
    """, (new_status, username))

    conn.commit()
    conn.close()


def get_user(username: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT username, password_hash, role, is_active
        FROM users
        WHERE username = ?
    """, (username,))

    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "username": row[0],
            "password_hash": row[1],
            "role": row[2],
            "is_active": bool(row[3])
        }
    return None


def get_all_users():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT username, role, is_active
        FROM users
        ORDER BY username
    """)

    rows = cur.fetchall()
    conn.close()

    return [{
        "username": r[0],
        "role": r[1],
        "is_active": bool(r[2])
    } for r in rows]


# -------------------------------------------------
# АУТЕНТИФИКАЦИЯ
# -------------------------------------------------

def authenticate(username: str, password: str):
    password_hash = hash_password(password)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT username, role
        FROM users
        WHERE username = ?
          AND password_hash = ?
          AND is_active = 1
    """, (username, password_hash))

    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "username": row[0],
            "role": row[1]
        }
    return None


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
# СТАТИСТИКА
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
# ЖУРНАЛ ДЕЙСТВИЙ АДМИНА
# -------------------------------------------------

def log_admin_action(admin: str, action: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO admin_log (admin, action, timestamp)
        VALUES (?, ?, ?)
    """, (
        admin,
        action,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_admin_logs():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT admin, action, timestamp
        FROM admin_log
        ORDER BY id DESC
    """)

    rows = cur.fetchall()
    conn.close()

    return [{
        "admin": r[0],
        "action": r[1],
        "timestamp": r[2]
    } for r in rows]
