#!/usr/bin/env python3
"""
Test script to check route registration
"""

from app import create_app

print("Creating app...")
app = create_app()

print("Registered routes:")
for rule in app.url_map.iter_rules():
    print(f"  {rule.rule} -> {rule.endpoint}")

print("\nTesting root route...")
with app.test_client() as client:
    response = client.get('/')
    print(f"Status: {response.status_code}")
    print(f"Data: {response.data.decode()[:100]}...")
