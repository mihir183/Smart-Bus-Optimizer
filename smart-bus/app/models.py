from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index

db = SQLAlchemy()

class Route(db.Model):
    """Bus route information"""
    __tablename__ = 'routes'
    
    id = db.Column(db.Integer, primary_key=True)
    route_number = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    start_stop = db.Column(db.String(100), nullable=False)
    end_stop = db.Column(db.String(100), nullable=False)
    total_distance = db.Column(db.Float)  # in kilometers
    estimated_duration = db.Column(db.Integer)  # in minutes
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    buses = db.relationship('Bus', backref='route', lazy=True)
    trips = db.relationship('Trip', backref='route', lazy=True)
    stops = db.relationship('RouteStop', backref='route', lazy=True)
    
    def __repr__(self):
        return f'<Route {self.route_number}: {self.name}>'

class Bus(db.Model):
    """Individual bus information"""
    __tablename__ = 'buses'
    
    id = db.Column(db.Integer, primary_key=True)
    bus_number = db.Column(db.String(20), unique=True, nullable=False)
    license_plate = db.Column(db.String(20), unique=True, nullable=False)
    capacity = db.Column(db.Integer, nullable=False)
    model = db.Column(db.String(50))
    year = db.Column(db.Integer)
    route_id = db.Column(db.Integer, db.ForeignKey('routes.id'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    trips = db.relationship('Trip', backref='bus', lazy=True)
    events = db.relationship('Event', backref='bus', lazy=True)
    
    def __repr__(self):
        return f'<Bus {self.bus_number}>'

class RouteStop(db.Model):
    """Stops along a route"""
    __tablename__ = 'route_stops'
    
    id = db.Column(db.Integer, primary_key=True)
    route_id = db.Column(db.Integer, db.ForeignKey('routes.id'), nullable=False)
    stop_name = db.Column(db.String(100), nullable=False)
    stop_code = db.Column(db.String(20))
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    sequence_order = db.Column(db.Integer, nullable=False)
    estimated_time_from_start = db.Column(db.Integer)  # minutes from route start
    
    # Index for efficient queries
    __table_args__ = (
        Index('idx_route_sequence', 'route_id', 'sequence_order'),
    )
    
    def __repr__(self):
        return f'<RouteStop {self.stop_name} (Route {self.route_id})>'

class Trip(db.Model):
    """Individual bus trips"""
    __tablename__ = 'trips'
    
    id = db.Column(db.Integer, primary_key=True)
    route_id = db.Column(db.Integer, db.ForeignKey('routes.id'), nullable=False)
    bus_id = db.Column(db.Integer, db.ForeignKey('buses.id'), nullable=False)
    scheduled_start_time = db.Column(db.DateTime, nullable=False)
    actual_start_time = db.Column(db.DateTime)
    scheduled_end_time = db.Column(db.DateTime, nullable=False)
    actual_end_time = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='scheduled')  # scheduled, in_progress, completed, cancelled
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    events = db.relationship('Event', backref='trip', lazy=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_trip_route_time', 'route_id', 'scheduled_start_time'),
        Index('idx_trip_bus_time', 'bus_id', 'scheduled_start_time'),
    )
    
    def __repr__(self):
        return f'<Trip {self.id}: Bus {self.bus_id} on Route {self.route_id}>'

class Event(db.Model):
    """Real-time events (GPS updates, occupancy, delays, etc.)"""
    __tablename__ = 'events'
    
    id = db.Column(db.Integer, primary_key=True)
    trip_id = db.Column(db.Integer, db.ForeignKey('trips.id'), nullable=False)
    bus_id = db.Column(db.Integer, db.ForeignKey('buses.id'), nullable=False)
    event_type = db.Column(db.String(50), nullable=False)  # gps_update, occupancy_update, delay, breakdown
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    speed = db.Column(db.Float)  # km/h
    heading = db.Column(db.Float)  # degrees
    occupancy_count = db.Column(db.Integer)
    occupancy_percentage = db.Column(db.Float)
    delay_minutes = db.Column(db.Integer)
    stop_id = db.Column(db.Integer, db.ForeignKey('route_stops.id'))
    data = db.Column(db.JSON)  # Additional event-specific data
    
    # Relationships
    stop = db.relationship('RouteStop', backref='events')
    
    # Indexes for efficient queries
    __table_args__ = (
        Index('idx_event_trip_time', 'trip_id', 'timestamp'),
        Index('idx_event_bus_time', 'bus_id', 'timestamp'),
        Index('idx_event_type_time', 'event_type', 'timestamp'),
    )
    
    def __repr__(self):
        return f'<Event {self.event_type} at {self.timestamp}>'

class Prediction(db.Model):
    """ML model predictions"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    trip_id = db.Column(db.Integer, db.ForeignKey('trips.id'), nullable=False)
    stop_id = db.Column(db.Integer, db.ForeignKey('route_stops.id'), nullable=False)
    predicted_arrival_time = db.Column(db.DateTime, nullable=False)
    confidence_score = db.Column(db.Float)
    model_version = db.Column(db.String(50))
    features_used = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    trip = db.relationship('Trip', backref='predictions')
    stop = db.relationship('RouteStop', backref='predictions')
    
    def __repr__(self):
        return f'<Prediction for Trip {self.trip_id} at Stop {self.stop_id}>'

class ScheduleAdjustment(db.Model):
    """Schedule adjustments made by the system"""
    __tablename__ = 'schedule_adjustments'
    
    id = db.Column(db.Integer, primary_key=True)
    route_id = db.Column(db.Integer, db.ForeignKey('routes.id'), nullable=False)
    adjustment_type = db.Column(db.String(50), nullable=False)  # bunching, delay, breakdown
    affected_trips = db.Column(db.JSON)  # List of trip IDs
    original_schedule = db.Column(db.JSON)
    adjusted_schedule = db.Column(db.JSON)
    reason = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    applied_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='pending')  # pending, applied, reverted
    
    # Relationships
    route = db.relationship('Route', backref='schedule_adjustments')
    
    def __repr__(self):
        return f'<ScheduleAdjustment {self.adjustment_type} for Route {self.route_id}>'
