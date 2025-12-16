import os
import datetime
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user, current_user, UserMixin
)

from openai import OpenAI

# 🧠 Импорт локальной модели
from ml_model import load_local_model, predict_local_feedback


# ================== НАСТРОЙКИ ==================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FLASK_SECRET = os.getenv("FLASK_SECRET", "секрет_123")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__, static_folder='static')
app.secret_key = FLASK_SECRET
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# ================== МОДЕЛИ БД ==================

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)


class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    username = db.Column(db.String(150), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    task = db.Column(db.Text)
    solution = db.Column(db.Text)
    feedback = db.Column(db.Text)


with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ================== ML МОДЕЛЬ ==================

# Загружается один раз при старте приложения
local_model = load_local_model()


# ================== OPENAI ==================

def generate_task():
    prompt = "Придумай учебную задачу по Python с функцией и примером её вызова."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, str(e)


def check_solution(task, solution, use_local_model=False):
    if use_local_model:
        try:
            feedback = predict_local_feedback(local_model, task, solution)
            return feedback, None
        except Exception as e:
            return None, f"Ошибка локальной модели: {str(e)}"
    else:
        prompt = (
            f"Вот задание:\n{task}\n\n"
            f"Вот код:\n```python\n{solution}\n```\n\n"
            "Скажи, правильно ли решено и что можно улучшить."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            return response.choices[0].message.content.strip(), None
        except Exception as e:
            return None, str(e)


# ================== РОУТЫ ==================

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    task = solution = feedback = error = None

    if request.method == 'POST':
        action = request.form.get('action')
        solution = request.form.get('solution', '').strip()

        if action == 'generate':
            task, err = generate_task()
            if err:
                error = err

        elif action == 'check':
            task = request.form.get('task', '')
            use_local = request.form.get('use_local_model') == 'on'

            if not task:
                error = 'Задание отсутствует.'
            elif not solution:
                error = 'Поле с решением пустое.'
            else:
                feedback, err = check_solution(task, solution, use_local_model=use_local)
                if err:
                    error = err
                else:
                    report = Report(
                        user_id=current_user.id,
                        username=current_user.username,
                        task=task,
                        solution=solution,
                        feedback=feedback
                    )
                    db.session.add(report)
                    db.session.commit()

    return render_template(
        'index.html',
        task=task,
        solution=solution,
        feedback=feedback,
        error_msg=error
    )


@app.route('/help')
@login_required
def help_page():
    return render_template('help.html')


@app.route('/admin')
@login_required
def admin_panel():
    if current_user.role != 'admin':
        flash('Доступ запрещён', 'danger')
        return redirect(url_for('index'))
    users = User.query.all()
    return render_template('admin.html', users=users)


@app.route('/user')
@login_required
def user_panel():
    return render_template('user.html')


@app.route('/teacher')
@login_required
def teacher_panel():
    if current_user.role != 'teacher':
        flash('Доступ запрещён', 'danger')
        return redirect(url_for('index'))
    reports = Report.query.order_by(Report.timestamp.desc()).all()
    return render_template('teacher.html', reports=reports)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(
            username=request.form['username'],
            password=request.form['password']
        ).first()
        if user:
            login_user(user)
            return redirect(url_for('index'))
        flash('Неверные данные', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        role = request.form['role']
        if User.query.filter_by(username=uname).first():
            flash('Пользователь уже есть', 'danger')
        else:
            db.session.add(User(username=uname, password=pwd, role=role))
            db.session.commit()
            flash('Успешно!', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
