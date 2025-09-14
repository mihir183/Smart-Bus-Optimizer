#!/usr/bin/env python3
"""
Simple test to check if the app can start
"""

print("Testing Smart Bus System startup...")

try:
    print("1. Testing imports...")
    from app import create_app, get_socketio
    print("✅ Imports successful")
    
    print("2. Creating Flask app...")
    app = create_app()
    print("✅ Flask app created")
    
    print("3. Getting SocketIO...")
    socketio = get_socketio()
    print("✅ SocketIO initialized")
    
    print("4. Testing app context...")
    with app.app_context():
        print("✅ App context works")
    
    print("\n🎉 All tests passed! The app should be able to run.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
