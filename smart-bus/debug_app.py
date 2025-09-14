#!/usr/bin/env python3
"""
Debug script to test app initialization
"""

import sys
import os

print("Starting debug...")

try:
    print("1. Importing Flask...")
    from flask import Flask
    print("   ✓ Flask imported successfully")
    
    print("2. Importing SQLAlchemy...")
    from flask_sqlalchemy import SQLAlchemy
    print("   ✓ SQLAlchemy imported successfully")
    
    print("3. Importing SocketIO...")
    from flask_socketio import SocketIO
    print("   ✓ SocketIO imported successfully")
    
    print("4. Importing CORS...")
    from flask_cors import CORS
    print("   ✓ CORS imported successfully")
    
    print("5. Creating basic Flask app...")
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    print("   ✓ Basic Flask app created")
    
    print("6. Initializing extensions...")
    db = SQLAlchemy()
    socketio = SocketIO()
    db.init_app(app)
    socketio.init_app(app)
    CORS(app)
    print("   ✓ Extensions initialized")
    
    print("7. Testing app context...")
    with app.app_context():
        print("   ✓ App context works")
    
    print("8. Testing route...")
    @app.route('/')
    def test():
        return "Hello World!"
    print("   ✓ Route added")
    
    print("9. Testing app run...")
    print("   App should be working now!")
    
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
