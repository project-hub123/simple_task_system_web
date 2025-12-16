import os
import datetime
from typing import List, Dict

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user, current_user, UserMixin
)

from openai import OpenAI
from ml_model import (
    load_local_model, retrain_model,
    predict_local_feedback, evaluate_model,
    get_model_stats
)

# ================= НАСТРОЙКИ =================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FLASK_SECRET = os.getenv("FLASK_SECRET", "secret_123")

client = OpenAI(api_key=OPENAI_API_KEY)

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

    admin = User.query.filter_by(username="admin").first()
    if not admin:
        db.session.add(User(
            username="admin",
            password="1234",
            role="admin"
        ))
        db.session.commit()
        print("SYSTEM: admin / 1234 создан")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================

def log_action(username: str, action: str) -> None:
    db.session.add(AuditLog(username=username, action=action))
    db.session.commit()

def get_user_reports(user_id: int) -> List[Report]:
    return Report.query.filter_by(user_id=user_id).order_by(Report.timestamp.desc()).all()

def get_system_stats() -> Dict[str, int]:
    return {
        "users": User.query.count(),
        "reports": Report.query.count(),
        "logs": AuditLog.query.count()
    }

# ================= ML =================

local_model = load_local_model()

# ================= OPENAI =================

def generate_task():
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Придумай учебную задачу по Python."}],
            max_tokens=150
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, str(e)

# ================= РОУТЫ =================

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    task = solution = feedback = error = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "generate":
            task, error = generate_task()
            log_action(current_user.username, "Сгенерировано задание")

        elif action == "check":
            task = request.form.get("task")
            solution = request.form.get("solution")
            use_local = request.form.get("use_local_model") == "on"

            if use_local:
                feedback = predict_local_feedback(local_model, task, solution)
            else:
                feedback, error = generate_task()

            if not error:
                db.session.add(Report(
                    user_id=current_user.id,
                    username=current_user.username,
                    task=task,
                    solution=solution,
                    feedback=feedback
                ))
                db.session.commit()
                log_action(current_user.username, "Проверено решение")

    return render_template(
        "index.html",
        task=task,
        solution=solution,
        feedback=feedback,
        error_msg=error
    )

@app.route("/model_stats")
@login_required
def model_stats():
    if current_user.role != "admin":
        return redirect(url_for("index"))

    stats = get_model_stats()
    metrics = evaluate_model(local_model)
    system = get_system_stats()

    return render_template(
        "model_stats.html",
        stats=stats,
        metrics=metrics,
        system=system
    )

@app.route("/retrain", methods=["POST"])
@login_required
def retrain():
    if current_user.role != "admin":
        return redirect(url_for("index"))

    global local_model
    local_model = retrain_model()
    log_action(current_user.username, "Переобучение модели")
    flash("Модель переобучена", "success")
    return redirect(url_for("model_stats"))

@app.route("/upload_dataset", methods=["GET", "POST"])
@login_required
def upload_dataset():
    if current_user.role != "admin":
        return redirect(url_for("index"))

    if request.method == "POST":
        file = request.files.get("dataset")
        if file:
            file.save("python_task_dataset.csv")
            log_action(current_user.username, "Загружен датасет")
            flash("Датасет загружен", "success")

    return render_template("upload_dataset.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        role = request.form.get("role", "user")

        if User.query.filter_by(username=username).first():
            flash("Пользователь уже существует")
            return redirect(url_for("register"))

        db.session.add(User(username=username, password=password, role=role))
        db.session.commit()
        log_action(username, "Регистрация пользователя")
        flash("Регистрация успешна")
        return redirect(url_for("login"))

    return render_template("register.html")

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
        flash("Ошибка входа")

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    log_action(current_user.username, "Выход из системы")
    logout_user()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
