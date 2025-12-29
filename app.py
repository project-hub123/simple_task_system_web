import sys
print("APP STARTED, PYTHON:", sys.executable)

import os
import datetime
from typing import Dict

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, session
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user, current_user, UserMixin
)

# ===== ИМПОРТ ИЗ ml (ЯВНО И КОРРЕКТНО) =====
from ml.task_generator import generate_task
from ml.predict import predict

# ================= НАСТРОЙКИ =================

FLASK_SECRET = os.getenv("FLASK_SECRET", "secret_123")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.secret_key = FLASK_SECRET

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(BASE_DIR, "db.sqlite3")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ================= МОДЕЛИ БД =================

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)


class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    username = db.Column(db.String(150))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    task_text = db.Column(db.Text)
    solution = db.Column(db.Text)
    feedback = db.Column(db.Text)


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150))
    action = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# ================= ИНИЦИАЛИЗАЦИЯ =================

with app.app_context():
    db.create_all()

    if not User.query.filter_by(username="admin").first():
        db.session.add(
            User(username="admin", password="1234", role="admin")
        )
        db.session.commit()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= ВСПОМОГАТЕЛЬНЫЕ =================

def log_action(username: str, action: str):
    db.session.add(AuditLog(username=username, action=action))
    db.session.commit()

def get_system_stats() -> Dict[str, int]:
    return {
        "users": User.query.count(),
        "reports": Report.query.count(),
        "logs": AuditLog.query.count()
    }

# ================= ГЛАВНАЯ СТРАНИЦА =================

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    task_text = ""
    solution = ""
    feedback = ""
    error = None

    if request.method == "POST":
        action = request.form.get("action")

        # ===== ГЕНЕРАЦИЯ ЗАДАНИЯ =====
        if action == "generate":
            try:
                task = generate_task()
                session["current_task"] = task
                task_text = task.get("task_text", "")
                log_action(current_user.username, "Сгенерировано задание")
            except Exception as e:
                error = f"Ошибка генерации: {e}"

        # ===== ПРОВЕРКА РЕШЕНИЯ =====
        elif action == "check":
            solution = request.form.get("solution", "")
            task = session.get("current_task")

            if not task:
                error = "Сначала сгенерируйте задание."
            elif not solution.strip():
                error = "Решение не может быть пустым."
            else:
                try:
                    feedback = predict(task, solution)

                    db.session.add(
                        Report(
                            user_id=current_user.id,
                            username=current_user.username,
                            task_text=task.get("task_text", ""),
                            solution=solution,
                            feedback=feedback
                        )
                    )
                    db.session.commit()

                    log_action(current_user.username, "Проверено решение")

                except Exception as e:
                    error = f"Ошибка проверки: {e}"

            task_text = task.get("task_text", "") if task else ""

    return render_template(
        "index.html",
        task=task_text,
        solution=solution,
        feedback=feedback,
        error_msg=error
    )

# ================= АВТОРИЗАЦИЯ =================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(
            username=request.form.get("username"),
            password=request.form.get("password")
        ).first()

        if user:
            login_user(user)
            log_action(user.username, "Вход в систему")
            return redirect(url_for("index"))

        flash("Ошибка входа", "danger")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        role = request.form.get("role", "user")

        if User.query.filter_by(username=username).first():
            flash("Пользователь уже существует", "danger")
            return redirect(url_for("register"))

        db.session.add(
            User(username=username, password=password, role=role)
        )
        db.session.commit()

        log_action(username, "Регистрация пользователя")
        flash("Регистрация успешна", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    log_action(current_user.username, "Выход из системы")
    logout_user()
    session.pop("current_task", None)
    return redirect(url_for("login"))

# ================= ПРОЧИЕ СТРАНИЦЫ =================

@app.route("/help")
@login_required
def help_page():
    return render_template("help.html")

@app.route("/admin")
@login_required
def admin_panel():
    if current_user.role != "admin":
        flash("Доступ запрещён", "danger")
        return redirect(url_for("index"))

    users = User.query.order_by(User.id).all()
    stats = get_system_stats()
    return render_template("admin.html", users=users, stats=stats)

@app.route("/teacher")
@login_required
def teacher_panel():
    if current_user.role != "teacher":
        flash("Доступ запрещён", "danger")
        return redirect(url_for("index"))

    return render_template("teacher.html")

@app.route("/user")
@login_required
def user_panel():
    return render_template("user.html")

# ================= ЛОКАЛЬНЫЙ ЗАПУСК =================

if __name__ == "__main__":
    app.run(debug=True)
