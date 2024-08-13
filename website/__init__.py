from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager

db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = "helloworld"
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    from .views import data_views  # Adjusted import to match the Blueprint name
    from .auth import auth

    app.register_blueprint(data_views, url_prefix="/")  # Register your data_views Blueprint
    app.register_blueprint(auth, url_prefix="/auth")   # Assuming 'auth' is another Blueprint

    from .models import User
    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app

def create_database(app=None):
    if not app:
        app = create_app()  # Create app if not provided

    with app.app_context():
        if not path.exists("website/" + DB_NAME):
            db.create_all()
            print("Created database!")
