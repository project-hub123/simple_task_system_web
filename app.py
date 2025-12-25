import sys
print("APP STARTED, PYTHON:", sys.executable)

from ml import generate_task
import os
import datetime
from typing import Dict

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user, current_user, UserMixin
)

# ВАЖНО: используем ТОЛЬКО predict
from ml.predict import predict

# ================= НАСТРОЙКИ =================

FLASK_SECRET = os.getenv("FLASK_SECRET", "secret_123")

app = Flask(__name__)
app.secret_key = FLASK_SECRET
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite3"
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
    task = db.Column(db.Text)
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
        db.session.add(User(
            username="admin",
            password="1234",
            role="admin"
        ))
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

# ================= РОУТЫ =================

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    task = ""
    solution = ""
    feedback = ""
    error = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "generate":
            task = generate_task()
            solution = ""
            feedback = ""
            error = None
            log_action(current_user.username, "Сгенерировано задание")

        elif action == "check":
            task = request.form.get("task", "")
            solution = request.form.get("solution", "")

            if not task.strip():
                error = "Задание отсутствует."
            elif not solution.strip():
                error = "Решение не может быть пустым."
            else:
                try:
                    feedback = predict(solution_text=solution, task_text=task)
                    error = None

                    db.session.add(Report(
                        user_id=current_user.id,
                        username=current_user.username,
                        task=task,
                        solution=solution,
                        feedback=feedback
                    ))
                    db.session.commit()

                    log_action(current_user.username, "Проверено решение")

                except Exception as e:
                    error = f"Ошибка проверки: {e}"
                    feedback = ""

    return render_template(
        "index.html",
        task=task,
        solution=solution,
        feedback=feedback,
        error_msg=error
    )

# ================= АВТОРИЗАЦИЯ =================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(
            username=request.form["username"],
            password=request.form["password"]
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

        db.session.add(User(
            username=username,
            password=password,
            role=role
        ))
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
    return redirect(url_for("login"))

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

# ================= ЗАПУСК =================

if __name__ == "__main__":
    app.run(debug=True)
