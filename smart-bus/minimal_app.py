#!/usr/bin/env python3
"""
Minimal Flask app to test basic functionality
"""

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World! Smart Bus System is working!"

@app.route('/test')
def test():
    return "Test route is working!"

if __name__ == '__main__':
    print("Starting minimal Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)
