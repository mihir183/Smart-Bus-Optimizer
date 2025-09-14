from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
from sqlalchemy import and_, or_, desc, func
from .models import db, Route, Bus, Trip, Event, Prediction, RouteStop, ScheduleAdjustment
from .predictor import Predictor
from .scheduler import Scheduler
from .utils import calculate_distance, format_time
import json

api_bp = Blueprint('api', __name__)

# Initialize services
predictor = Predictor()
scheduler = Scheduler()

@api_bp.route('/routes', methods=['GET'])
def get_routes():
    """Get all active routes"""
    try:
        routes = Route.query.filter_by(is_active=True).all()
        return jsonify({
            'success': True,
            'data': [{
                'id': route.id,
                'route_number': route.route_number,
                'name': route.name,
                'description': route.description,
                'start_stop': route.start_stop,
                'end_stop': route.end_stop,
                'total_distance': route.total_distance,
                'estimated_duration': route.estimated_duration,
                'created_at': route.created_at.isoformat()
            } for route in routes]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/routes/<int:route_id>', methods=['GET'])
def get_route(route_id):
    """Get specific route with stops"""
    try:
        route = Route.query.get_or_404(route_id)
        stops = RouteStop.query.filter_by(route_id=route_id).order_by(RouteStop.sequence_order).all()
        
        return jsonify({
            'success': True,
            'data': {
                'route': {
                    'id': route.id,
                    'route_number': route.route_number,
                    'name': route.name,
                    'description': route.description,
                    'start_stop': route.start_stop,
                    'end_stop': route.end_stop,
                    'total_distance': route.total_distance,
                    'estimated_duration': route.estimated_duration
                },
                'stops': [{
                    'id': stop.id,
                    'name': stop.stop_name,
                    'code': stop.stop_code,
                    'latitude': stop.latitude,
                    'longitude': stop.longitude,
                    'sequence_order': stop.sequence_order,
                    'estimated_time_from_start': stop.estimated_time_from_start
                } for stop in stops]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/buses', methods=['GET'])
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

@api_bp.route('/trips', methods=['GET'])
def get_trips():
    """Get trips with optional filters"""
    try:
        route_id = request.args.get('route_id', type=int)
        bus_id = request.args.get('bus_id', type=int)
        date = request.args.get('date')
        status = request.args.get('status')
        
        query = Trip.query
        
        if route_id:
            query = query.filter_by(route_id=route_id)
        if bus_id:
            query = query.filter_by(bus_id=bus_id)
        if date:
            date_obj = datetime.strptime(date, '%Y-%m-%d').date()
            query = query.filter(func.date(Trip.scheduled_start_time) == date_obj)
        if status:
            query = query.filter_by(status=status)
            
        trips = query.order_by(Trip.scheduled_start_time).all()
        
        return jsonify({
            'success': True,
            'data': [{
                'id': trip.id,
                'route_id': trip.route_id,
                'route_number': trip.route.route_number if trip.route else None,
                'bus_id': trip.bus_id,
                'bus_number': trip.bus.bus_number if trip.bus else None,
                'scheduled_start_time': trip.scheduled_start_time.isoformat(),
                'actual_start_time': trip.actual_start_time.isoformat() if trip.actual_start_time else None,
                'scheduled_end_time': trip.scheduled_end_time.isoformat(),
                'actual_end_time': trip.actual_end_time.isoformat() if trip.actual_end_time else None,
                'status': trip.status
            } for trip in trips]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/events', methods=['GET'])
def get_events():
    """Get recent events with optional filters"""
    try:
        trip_id = request.args.get('trip_id', type=int)
        bus_id = request.args.get('bus_id', type=int)
        event_type = request.args.get('event_type')
        hours = request.args.get('hours', 24, type=int)
        
        since_time = datetime.utcnow() - timedelta(hours=hours)
        query = Event.query.filter(Event.timestamp >= since_time)
        
        if trip_id:
            query = query.filter_by(trip_id=trip_id)
        if bus_id:
            query = query.filter_by(bus_id=bus_id)
        if event_type:
            query = query.filter_by(event_type=event_type)
            
        events = query.order_by(desc(Event.timestamp)).limit(1000).all()
        
        return jsonify({
            'success': True,
            'data': [{
                'id': event.id,
                'trip_id': event.trip_id,
                'bus_id': event.bus_id,
                'event_type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'latitude': event.latitude,
                'longitude': event.longitude,
                'speed': event.speed,
                'heading': event.heading,
                'occupancy_count': event.occupancy_count,
                'occupancy_percentage': event.occupancy_percentage,
                'delay_minutes': event.delay_minutes,
                'stop_id': event.stop_id,
                'data': event.data
            } for event in events]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/predictions', methods=['GET'])
def get_predictions():
    """Get arrival time predictions"""
    try:
        route_id = request.args.get('route_id', type=int)
        stop_id = request.args.get('stop_id', type=int)
        hours_ahead = request.args.get('hours_ahead', 2, type=int)
        
        since_time = datetime.utcnow()
        until_time = since_time + timedelta(hours=hours_ahead)
        
        query = Prediction.query.filter(
            and_(
                Prediction.predicted_arrival_time >= since_time,
                Prediction.predicted_arrival_time <= until_time
            )
        )
        
        if route_id:
            query = query.join(Trip).filter(Trip.route_id == route_id)
        if stop_id:
            query = query.filter_by(stop_id=stop_id)
            
        predictions = query.order_by(Prediction.predicted_arrival_time).all()
        
        return jsonify({
            'success': True,
            'data': [{
                'id': pred.id,
                'trip_id': pred.trip_id,
                'stop_id': pred.stop_id,
                'stop_name': pred.stop.stop_name if pred.stop else None,
                'predicted_arrival_time': pred.predicted_arrival_time.isoformat(),
                'confidence_score': pred.confidence_score,
                'model_version': pred.model_version,
                'created_at': pred.created_at.isoformat()
            } for pred in predictions]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/predictions/generate', methods=['POST'])
def generate_predictions():
    """Generate new predictions for a route"""
    try:
        data = request.get_json()
        route_id = data.get('route_id')
        hours_ahead = data.get('hours_ahead', 2)
        
        if not route_id:
            return jsonify({'success': False, 'error': 'route_id is required'}), 400
            
        # Get upcoming trips for the route
        since_time = datetime.utcnow()
        until_time = since_time + timedelta(hours=hours_ahead)
        
        trips = Trip.query.filter(
            and_(
                Trip.route_id == route_id,
                Trip.scheduled_start_time >= since_time,
                Trip.scheduled_start_time <= until_time,
                Trip.status.in_(['scheduled', 'in_progress'])
            )
        ).all()
        
        predictions = []
        for trip in trips:
            # Generate predictions for each stop on the route
            stops = RouteStop.query.filter_by(route_id=route_id).order_by(RouteStop.sequence_order).all()
            
            for stop in stops:
                prediction = predictor.predict_arrival_time(trip, stop)
                if prediction:
                    predictions.append(prediction)
        
        return jsonify({
            'success': True,
            'data': {
                'predictions_generated': len(predictions),
                'route_id': route_id,
                'hours_ahead': hours_ahead
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/schedule/adjustments', methods=['GET'])
def get_schedule_adjustments():
    """Get recent schedule adjustments"""
    try:
        route_id = request.args.get('route_id', type=int)
        days = request.args.get('days', 7, type=int)
        
        since_time = datetime.utcnow() - timedelta(days=days)
        query = ScheduleAdjustment.query.filter(ScheduleAdjustment.created_at >= since_time)
        
        if route_id:
            query = query.filter_by(route_id=route_id)
            
        adjustments = query.order_by(desc(ScheduleAdjustment.created_at)).all()
        
        return jsonify({
            'success': True,
            'data': [{
                'id': adj.id,
                'route_id': adj.route_id,
                'route_number': adj.route.route_number if adj.route else None,
                'adjustment_type': adj.adjustment_type,
                'affected_trips': adj.affected_trips,
                'reason': adj.reason,
                'status': adj.status,
                'created_at': adj.created_at.isoformat(),
                'applied_at': adj.applied_at.isoformat() if adj.applied_at else None
            } for adj in adjustments]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/schedule/detect-bunching', methods=['POST'])
def detect_bunching():
    """Detect and handle bus bunching"""
    try:
        data = request.get_json()
        route_id = data.get('route_id')
        
        if not route_id:
            return jsonify({'success': False, 'error': 'route_id is required'}), 400
            
        bunching_events = scheduler.detect_bunching(route_id)
        
        return jsonify({
            'success': True,
            'data': {
                'bunching_detected': len(bunching_events) > 0,
                'bunching_events': bunching_events,
                'route_id': route_id
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/schedule/reschedule', methods=['POST'])
def reschedule():
    """Apply schedule adjustments"""
    try:
        data = request.get_json()
        route_id = data.get('route_id')
        adjustment_type = data.get('adjustment_type', 'bunching')
        
        if not route_id:
            return jsonify({'success': False, 'error': 'route_id is required'}), 400
            
        adjustment = scheduler.create_schedule_adjustment(route_id, adjustment_type)
        
        return jsonify({
            'success': True,
            'data': {
                'adjustment_id': adjustment.id,
                'route_id': route_id,
                'adjustment_type': adjustment_type,
                'status': adjustment.status
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/analytics/route-performance', methods=['GET'])
def get_route_performance():
    """Get route performance analytics"""
    try:
        route_id = request.args.get('route_id', type=int)
        days = request.args.get('days', 7, type=int)
        
        since_time = datetime.utcnow() - timedelta(days=days)
        
        # Get trips for the period
        query = Trip.query.filter(Trip.scheduled_start_time >= since_time)
        if route_id:
            query = query.filter_by(route_id=route_id)
            
        trips = query.all()
        
        # Calculate metrics
        total_trips = len(trips)
        on_time_trips = len([t for t in trips if t.actual_start_time and 
                           abs((t.actual_start_time - t.scheduled_start_time).total_seconds()) < 300])
        
        on_time_percentage = (on_time_trips / total_trips * 100) if total_trips > 0 else 0
        
        # Get average delays
        delay_events = Event.query.filter(
            and_(
                Event.event_type == 'delay',
                Event.timestamp >= since_time
            )
        )
        if route_id:
            delay_events = delay_events.join(Trip).filter(Trip.route_id == route_id)
            
        delays = [e.delay_minutes for e in delay_events if e.delay_minutes]
        avg_delay = sum(delays) / len(delays) if delays else 0
        
        return jsonify({
            'success': True,
            'data': {
                'route_id': route_id,
                'period_days': days,
                'total_trips': total_trips,
                'on_time_trips': on_time_trips,
                'on_time_percentage': round(on_time_percentage, 2),
                'average_delay_minutes': round(avg_delay, 2),
                'total_delays': len(delays)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/analytics/occupancy', methods=['GET'])
def get_occupancy_analytics():
    """Get occupancy analytics"""
    try:
        route_id = request.args.get('route_id', type=int)
        hours = request.args.get('hours', 24, type=int)
        
        since_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = Event.query.filter(
            and_(
                Event.event_type == 'occupancy_update',
                Event.timestamp >= since_time,
                Event.occupancy_percentage.isnot(None)
            )
        )
        
        if route_id:
            query = query.join(Trip).filter(Trip.route_id == route_id)
            
        occupancy_events = query.all()
        
        if not occupancy_events:
            return jsonify({
                'success': True,
                'data': {
                    'route_id': route_id,
                    'period_hours': hours,
                    'average_occupancy': 0,
                    'peak_occupancy': 0,
                    'total_updates': 0
                }
            })
        
        occupancies = [e.occupancy_percentage for e in occupancy_events]
        avg_occupancy = sum(occupancies) / len(occupancies)
        peak_occupancy = max(occupancies)
        
        return jsonify({
            'success': True,
            'data': {
                'route_id': route_id,
                'period_hours': hours,
                'average_occupancy': round(avg_occupancy, 2),
                'peak_occupancy': round(peak_occupancy, 2),
                'total_updates': len(occupancy_events)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'database': 'connected',
                'predictor': 'available',
                'scheduler': 'available'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
