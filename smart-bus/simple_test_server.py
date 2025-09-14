#!/usr/bin/env python3
"""
Simple test server to verify Flask is working
"""

from flask import Blueprint, request, jsonify, current_app
api = Blueprint('api', __name__)


@api.route('/')
def hello():    
    return '<h1>Smart Bus System is Working!</h1><p>If you can see this, the server is running correctly.</p>'

@api.route('/api/health')
def health():
    return {'status': 'ok', 'message': 'Smart Bus System is running'}

@api.route('/buses', methods=['GET'])
def get_buses():
    """Get all active buses"""
    try:
        buses = Bus.query.filter_by(is_active=True).all()
        return jsonify({
            'success': True,
            'data': [{
                'id': bus.id,
                'bus_number': bus.bus_number,
                'license_plate': bus.license_plate,
                'capacity': bus.capacity,
                'model': bus.model,
                'year': bus.year,
                'route_id': bus.route_id,
                'route_number': bus.route.route_number if bus.route else None
            } for bus in buses]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting simple test server on http://localhost:5000")
    api.run(host='127.0.0.1', port=5000, debug=True)
