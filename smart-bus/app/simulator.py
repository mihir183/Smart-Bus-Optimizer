import random
import time
import threading
from datetime import datetime, timedelta
from sqlalchemy import and_, desc
from .models import db, Route, Bus, Trip, Event, RouteStop
from .utils import calculate_distance, interpolate_position
import logging

logger = logging.getLogger(__name__)

class BusSimulator:
    """Simulates real-time bus operations with GPS and occupancy updates"""
    
    def __init__(self, socketio=None):
        self.socketio = socketio
        self.running = False
        self.simulation_threads = {}
        self.update_interval = 30  # seconds between updates
        
        # Simulation parameters
        self.avg_speed_kmh = 25  # Average bus speed
        self.speed_variance = 0.2  # Speed variation factor
        self.occupancy_variance = 0.3  # Occupancy variation factor
        self.delay_probability = 0.1  # 10% chance of delay per update
        
    def start_simulation(self, route_id=None):
        """Start simulation for all routes or a specific route"""
        try:
            from flask import current_app
            
            with current_app.app_context():
                if route_id:
                    routes = Route.query.filter_by(id=route_id, is_active=True).all()
                else:
                    routes = Route.query.filter_by(is_active=True).all()
                
                if not routes:
                    logger.warning("No active routes found for simulation")
                    return
                
                self.running = True
                
                for route in routes:
                    if route.id not in self.simulation_threads:
                        thread = threading.Thread(
                            target=self._simulate_route,
                            args=(route.id,),
                            daemon=True
                        )
                        thread.start()
                        self.simulation_threads[route.id] = thread
                        logger.info(f"Started simulation for route {route.route_number}")
            
        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}")
    
    def stop_simulation(self, route_id=None):
        """Stop simulation for all routes or a specific route"""
        try:
            if route_id:
                if route_id in self.simulation_threads:
                    del self.simulation_threads[route_id]
                    logger.info(f"Stopped simulation for route {route_id}")
            else:
                self.running = False
                self.simulation_threads.clear()
                logger.info("Stopped all simulations")
                
        except Exception as e:
            logger.error(f"Error stopping simulation: {str(e)}")
    
    def _simulate_route(self, route_id):
        """Simulate a single route"""
        try:
            from flask import current_app
            
            with current_app.app_context():
                route = Route.query.get(route_id)
                if not route:
                    logger.error(f"Route {route_id} not found")
                    return
                
                # Get route stops
                stops = RouteStop.query.filter_by(route_id=route_id).order_by(RouteStop.sequence_order).all()
                if not stops:
                    logger.error(f"No stops found for route {route_id}")
                    return
                
                while self.running and route_id in self.simulation_threads:
                    try:
                        # Get active trips for this route
                        active_trips = self._get_active_trips(route_id)
                        
                        for trip in active_trips:
                            self._simulate_trip(trip, stops)
                        
                        # Wait before next update
                        time.sleep(self.update_interval)
                        
                    except Exception as e:
                        logger.error(f"Error in route simulation loop: {str(e)}")
                        time.sleep(self.update_interval)
                    
        except Exception as e:
            logger.error(f"Error simulating route {route_id}: {str(e)}")
    
    def _get_active_trips(self, route_id):
        """Get currently active trips for a route"""
        from flask import current_app
        
        with current_app.app_context():
            now = datetime.utcnow()
            time_window = timedelta(hours=2)  # Look 2 hours ahead and behind
            
            return Trip.query.filter(
                and_(
                    Trip.route_id == route_id,
                    Trip.status.in_(['scheduled', 'in_progress']),
                    Trip.scheduled_start_time >= now - time_window,
                    Trip.scheduled_start_time <= now + time_window
                )
            ).all()
    
    def _simulate_trip(self, trip, stops):
        """Simulate a single trip"""
        try:
            from flask import current_app
            
            with current_app.app_context():
                # Get latest GPS event for this trip
                latest_event = Event.query.filter(
                    and_(
                        Event.trip_id == trip.id,
                        Event.event_type == 'gps_update'
                    )
                ).order_by(desc(Event.timestamp)).first()
                
                # Determine current position and progress
                if latest_event:
                    current_position = (latest_event.latitude, latest_event.longitude)
                    current_stop_index = self._find_nearest_stop_index(current_position, stops)
                    progress = current_stop_index / len(stops) if stops else 0
                else:
                    # Trip just started, begin from first stop
                    current_position = (stops[0].latitude, stops[0].longitude)
                    current_stop_index = 0
                    progress = 0
                    trip.status = 'in_progress'
                    if not trip.actual_start_time:
                        trip.actual_start_time = datetime.utcnow()
                
                # Calculate next position
                next_position = self._calculate_next_position(trip, stops, current_stop_index, progress)
                
                # Generate GPS update
                self._generate_gps_update(trip, next_position)
                
                # Generate occupancy update
                self._generate_occupancy_update(trip)
                
                # Check for delays
                if random.random() < self.delay_probability:
                    self._generate_delay_event(trip)
                
                # Update trip status
                if progress >= 0.95:  # Trip almost complete
                    trip.status = 'completed'
                    if not trip.actual_end_time:
                        trip.actual_end_time = datetime.utcnow()
                
                db.session.commit()
            
        except Exception as e:
            logger.error(f"Error simulating trip {trip.id}: {str(e)}")
            from flask import current_app
            with current_app.app_context():
                db.session.rollback()
    
    def _find_nearest_stop_index(self, position, stops):
        """Find the index of the nearest stop to current position"""
        min_distance = float('inf')
        nearest_index = 0
        
        for i, stop in enumerate(stops):
            distance = calculate_distance(position[0], position[1], stop.latitude, stop.longitude)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        
        return nearest_index
    
    def _calculate_next_position(self, trip, stops, current_stop_index, progress):
        """Calculate the next GPS position for a bus"""
        try:
            # Determine target stop
            if current_stop_index < len(stops) - 1:
                target_stop = stops[current_stop_index + 1]
            else:
                # At or near end of route
                target_stop = stops[-1]
            
            # Get current position
            latest_event = Event.query.filter(
                and_(
                    Event.trip_id == trip.id,
                    Event.event_type == 'gps_update'
                )
            ).order_by(desc(Event.timestamp)).first()
            
            if latest_event:
                current_lat, current_lon = latest_event.latitude, latest_event.longitude
            else:
                current_lat, current_lon = stops[current_stop_index].latitude, stops[current_stop_index].longitude
            
            # Calculate movement based on speed and time
            speed_kmh = self._calculate_current_speed(trip, progress)
            distance_km = speed_kmh * (self.update_interval / 3600)  # Distance in this update interval
            
            # Move towards target stop
            next_lat, next_lon = self._move_towards_target(
                current_lat, current_lon,
                target_stop.latitude, target_stop.longitude,
                distance_km
            )
            
            return (next_lat, next_lon)
            
        except Exception as e:
            logger.error(f"Error calculating next position: {str(e)}")
            # Return current position if calculation fails
            return (stops[current_stop_index].latitude, stops[current_stop_index].longitude)
    
    def _calculate_current_speed(self, trip, progress):
        """Calculate current bus speed based on various factors"""
        base_speed = self.avg_speed_kmh
        
        # Add some randomness
        speed_variation = random.uniform(-self.speed_variance, self.speed_variance)
        current_speed = base_speed * (1 + speed_variation)
        
        # Slow down near stops
        if progress < 0.1 or progress > 0.9:  # Beginning or end of route
            current_speed *= 0.7
        
        # Add traffic simulation (random delays)
        if random.random() < 0.05:  # 5% chance of traffic
            current_speed *= 0.5
        
        return max(current_speed, 5)  # Minimum speed of 5 km/h
    
    def _move_towards_target(self, current_lat, current_lon, target_lat, target_lon, distance_km):
        """Move from current position towards target by specified distance"""
        # Calculate bearing to target
        bearing = self._calculate_bearing(current_lat, current_lon, target_lat, target_lon)
        
        # Calculate new position
        new_lat, new_lon = self._move_by_bearing(current_lat, current_lon, bearing, distance_km)
        
        return new_lat, new_lon
    
    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate bearing between two points"""
        import math
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def _move_by_bearing(self, lat, lon, bearing, distance_km):
        """Move from a point by bearing and distance"""
        import math
        
        earth_radius_km = 6371
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)
        
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance_km / earth_radius_km) +
            math.cos(lat_rad) * math.sin(distance_km / earth_radius_km) * math.cos(bearing_rad)
        )
        
        new_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(distance_km / earth_radius_km) * math.cos(lat_rad),
            math.cos(distance_km / earth_radius_km) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        return math.degrees(new_lat_rad), math.degrees(new_lon_rad)
    
    def _generate_gps_update(self, trip, position):
        """Generate a GPS update event"""
        try:
            from flask import current_app
            
            with current_app.app_context():
                # Calculate speed and heading
                latest_event = Event.query.filter(
                    and_(
                        Event.trip_id == trip.id,
                        Event.event_type == 'gps_update'
                    )
                ).order_by(desc(Event.timestamp)).first()
            
            speed = 0
            heading = 0
            
            if latest_event:
                # Calculate speed based on distance and time
                distance = calculate_distance(
                    latest_event.latitude, latest_event.longitude,
                    position[0], position[1]
                )
                time_diff = (datetime.utcnow() - latest_event.timestamp).total_seconds()
                if time_diff > 0:
                    speed = (distance / time_diff) * 3600  # km/h
                
                # Calculate heading
                heading = self._calculate_bearing(
                    latest_event.latitude, latest_event.longitude,
                    position[0], position[1]
                )
            
            # Create GPS event
            gps_event = Event(
                trip_id=trip.id,
                bus_id=trip.bus_id,
                event_type='gps_update',
                timestamp=datetime.utcnow(),
                latitude=position[0],
                longitude=position[1],
                speed=speed,
                heading=heading
            )
            
            db.session.add(gps_event)
            
            # Emit real-time update via SocketIO
            if self.socketio:
                self.socketio.emit('gps_update', {
                    'trip_id': trip.id,
                    'bus_id': trip.bus_id,
                    'route_id': trip.route_id,
                    'latitude': position[0],
                    'longitude': position[1],
                    'speed': speed,
                    'heading': heading,
                    'timestamp': gps_event.timestamp.isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error generating GPS update: {str(e)}")
    
    def _generate_occupancy_update(self, trip):
        """Generate an occupancy update event"""
        try:
            # Simulate realistic occupancy patterns
            bus = Bus.query.get(trip.bus_id)
            if not bus:
                return
            
            capacity = bus.capacity
            
            # Base occupancy varies by time of day and route progress
            now = datetime.utcnow()
            hour = now.hour
            
            # Peak hours (7-9 AM, 5-7 PM) have higher occupancy
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                base_occupancy = 0.7
            elif 10 <= hour <= 16:
                base_occupancy = 0.4
            else:
                base_occupancy = 0.2
            
            # Add randomness
            occupancy_variation = random.uniform(-self.occupancy_variance, self.occupancy_variance)
            occupancy_percentage = max(0, min(1, base_occupancy + occupancy_variation))
            
            occupancy_count = int(capacity * occupancy_percentage)
            
            # Create occupancy event
            occupancy_event = Event(
                trip_id=trip.id,
                bus_id=trip.bus_id,
                event_type='occupancy_update',
                timestamp=datetime.utcnow(),
                occupancy_count=occupancy_count,
                occupancy_percentage=occupancy_percentage * 100
            )
            
            db.session.add(occupancy_event)
            
            # Emit real-time update via SocketIO
            if self.socketio:
                self.socketio.emit('occupancy_update', {
                    'trip_id': trip.id,
                    'bus_id': trip.bus_id,
                    'route_id': trip.route_id,
                    'occupancy_count': occupancy_count,
                    'occupancy_percentage': occupancy_percentage * 100,
                    'capacity': capacity,
                    'timestamp': occupancy_event.timestamp.isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error generating occupancy update: {str(e)}")
    
    def _generate_delay_event(self, trip):
        """Generate a delay event"""
        try:
            # Random delay between 1-10 minutes
            delay_minutes = random.randint(1, 10)
            
            delay_event = Event(
                trip_id=trip.id,
                bus_id=trip.bus_id,
                event_type='delay',
                timestamp=datetime.utcnow(),
                delay_minutes=delay_minutes,
                data={
                    'reason': random.choice(['traffic', 'passenger_loading', 'mechanical', 'weather']),
                    'location': 'en_route'
                }
            )
            
            db.session.add(delay_event)
            
            # Emit real-time update via SocketIO
            if self.socketio:
                self.socketio.emit('delay_update', {
                    'trip_id': trip.id,
                    'bus_id': trip.bus_id,
                    'route_id': trip.route_id,
                    'delay_minutes': delay_minutes,
                    'reason': delay_event.data.get('reason'),
                    'timestamp': delay_event.timestamp.isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error generating delay event: {str(e)}")
    
    def get_simulation_status(self):
        """Get current simulation status"""
        return {
            'running': self.running,
            'active_routes': list(self.simulation_threads.keys()),
            'update_interval': self.update_interval
        }
    
    def update_simulation_parameters(self, **kwargs):
        """Update simulation parameters"""
        if 'update_interval' in kwargs:
            self.update_interval = max(10, kwargs['update_interval'])  # Minimum 10 seconds
        
        if 'avg_speed_kmh' in kwargs:
            self.avg_speed_kmh = max(5, kwargs['avg_speed_kmh'])  # Minimum 5 km/h
        
        if 'speed_variance' in kwargs:
            self.speed_variance = max(0, min(1, kwargs['speed_variance']))  # 0-1 range
        
        if 'occupancy_variance' in kwargs:
            self.occupancy_variance = max(0, min(1, kwargs['occupancy_variance']))  # 0-1 range
        
        if 'delay_probability' in kwargs:
            self.delay_probability = max(0, min(1, kwargs['delay_probability']))  # 0-1 range
        
        logger.info(f"Updated simulation parameters: {kwargs}")
