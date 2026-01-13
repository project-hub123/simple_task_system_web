import hashlib

from ml.database import (
    init_db,
    get_connection
)

# -------------------------------------------------
# ВСПОМОГАТЕЛЬНОЕ
# -------------------------------------------------

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# -------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ
# -------------------------------------------------

def init_system():
    init_db()

    # пользователи по умолчанию
    register_user("student", "student", "student123")
    register_user("teacher", "teacher", "teacher123")
    register_user("admin", "admin", "admin123")


# -------------------------------------------------
# РЕГИСТРАЦИЯ
# -------------------------------------------------

def register_user(username: str, role: str, password: str):
    if not username or not password:
        raise ValueError("Пустые данные")

    if role not in ("student", "teacher", "admin"):
        raise ValueError("Некорректная роль")

    password_hash = hash_password(password)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO users (username, password_hash, role)
        VALUES (?, ?, ?)
    """, (username, password_hash, role))
    conn.commit()
    conn.close()


# -------------------------------------------------
# ВХОД
# -------------------------------------------------

def login(username: str, password: str):
    if not username or not password:
        return None

    password_hash = hash_password(password)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT username, role
        FROM users
        WHERE username = ? AND password_hash = ?
    """, (username, password_hash))

    row = cur.fetchone()
    conn.close()

    if row:
        return {"username": row[0], "role": row[1]}
    return None


# -------------------------------------------------
# СПИСОК ПОЛЬЗОВАТЕЛЕЙ (АДМИН)
# -------------------------------------------------

def list_users():
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
