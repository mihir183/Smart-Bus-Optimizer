from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from flask_cors import CORS
from sqlalchemy import text   # ✅ needed for raw SQL
from .api import api_bp
import os

# Initialize extensions
db = SQLAlchemy()
socketio = SocketIO()

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///smart_bus.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions with app
    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")
    CORS(app)
    
    # Create database tables
    with app.app_context():
        # Import models to ensure they are registered
        from . import models
        db.create_all()
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')

    # Add root route
    @app.route('/')
    def index():
        from flask import render_template
        return render_template('index.html')

    @app.route('/map')
    def map_view():
        from flask import render_template
        return render_template('map.html')

    @app.route('/routes')
    def routes_view():
        from flask import render_template
        return render_template('route.html')

    # ✅ Health check route (fixes the SELECT 1 error)
    @app.route('/health')
    def health():
        try:
            db.session.execute(text("SELECT 1"))
            return {"status": "healthy"}, 200
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}, 500

    return app
    return app
    
    # Add root route
    @app.route('/')
    def index():
        from flask import render_template
        return render_template('index.html')
    
    @app.route('/map')
    def map_view():
        from flask import render_template
        return render_template('map.html')
    
    @app.route('/routes')
    def routes_view():
        from flask import render_template
        return render_template('route.html')
    
    # ✅ Health check route (fixes the SELECT 1 error)
    @app.route('/health')
    def health():
        try:
            db.session.execute(text("SELECT 1"))
            return {"status": "healthy"}, 200
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}, 500

    return app

def get_socketio():
    """Get SocketIO instance for use in other modules"""
    return socketio
