from flask import Flask
from flask_cors import CORS
from app.routes import main

def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config.from_object('app.config.Config')

    # Register Blueprints (modular routes)
    app.register_blueprint(main)

    return app
