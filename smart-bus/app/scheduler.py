from datetime import datetime, timedelta
from sqlalchemy import and_, or_, desc, func
from .models import db, Route, Bus, Trip, Event, ScheduleAdjustment, RouteStop
from .utils import calculate_distance, format_time
import json
import logging

logger = logging.getLogger(__name__)

class Scheduler:
    """Intelligent scheduling engine for bus operations"""
    
    def __init__(self):
        self.bunching_threshold_minutes = 3  # Buses within 3 minutes are considered bunched
        self.min_headway_minutes = 5  # Minimum time between buses
        self.max_headway_minutes = 20  # Maximum time between buses
        
    def detect_bunching(self, route_id, time_window_minutes=30):
        """
        Detect bus bunching on a specific route
        
        Args:
            route_id (int): Route ID to check
            time_window_minutes (int): Time window to check for bunching
            
        Returns:
            list: List of bunching events detected
        """
        try:
            now = datetime.utcnow()
            time_window = timedelta(minutes=time_window_minutes)
            
            # Get active trips on the route
            active_trips = Trip.query.filter(
                and_(
                    Trip.route_id == route_id,
                    Trip.status == 'in_progress',
                    Trip.scheduled_start_time >= now - time_window,
                    Trip.scheduled_start_time <= now + time_window
                )
            ).order_by(Trip.scheduled_start_time).all()
            
            if len(active_trips) < 2:
                return []
            
            bunching_events = []
            
            # Check for bunching by comparing consecutive trips
            for i in range(len(active_trips) - 1):
                current_trip = active_trips[i]
                next_trip = active_trips[i + 1]
                
                # Get latest GPS events for both trips
                current_gps = self._get_latest_gps_event(current_trip.id)
                next_gps = self._get_latest_gps_event(next_trip.id)
                
                if current_gps and next_gps:
                    # Calculate distance between buses
                    distance = calculate_distance(
                        current_gps.latitude, current_gps.longitude,
                        next_gps.latitude, next_gps.longitude
                    )
                    
                    # Calculate time difference
                    time_diff = abs((current_gps.timestamp - next_gps.timestamp).total_seconds() / 60)
                    
                    # Check if buses are too close in time and space
                    if (time_diff <= self.bunching_threshold_minutes and 
                        distance <= 1.0):  # Within 1 km
                        
                        bunching_events.append({
                            'trip_1_id': current_trip.id,
                            'trip_2_id': next_trip.id,
                            'bus_1_id': current_trip.bus_id,
                            'bus_2_id': next_trip.bus_id,
                            'distance_km': round(distance, 2),
                            'time_diff_minutes': round(time_diff, 2),
                            'detected_at': now.isoformat(),
                            'severity': self._calculate_bunching_severity(time_diff, distance)
                        })
            
            return bunching_events
            
        except Exception as e:
            logger.error(f"Error detecting bunching for route {route_id}: {str(e)}")
            return []
    
    def create_schedule_adjustment(self, route_id, adjustment_type, reason=None):
        """
        Create a schedule adjustment to resolve issues
        
        Args:
            route_id (int): Route ID
            adjustment_type (str): Type of adjustment (bunching, delay, breakdown)
            reason (str): Reason for adjustment
            
        Returns:
            ScheduleAdjustment: Created adjustment record
        """
        try:
            # Get current schedule for the route
            now = datetime.utcnow()
            future_trips = Trip.query.filter(
                and_(
                    Trip.route_id == route_id,
                    Trip.scheduled_start_time >= now,
                    Trip.status.in_(['scheduled', 'in_progress'])
                )
            ).order_by(Trip.scheduled_start_time).all()
            
            if not future_trips:
                raise ValueError("No future trips found for route")
            
            # Create adjustment based on type
            if adjustment_type == 'bunching':
                adjustment = self._handle_bunching_adjustment(route_id, future_trips, reason)
            elif adjustment_type == 'delay':
                adjustment = self._handle_delay_adjustment(route_id, future_trips, reason)
            elif adjustment_type == 'breakdown':
                adjustment = self._handle_breakdown_adjustment(route_id, future_trips, reason)
            else:
                raise ValueError(f"Unknown adjustment type: {adjustment_type}")
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error creating schedule adjustment: {str(e)}")
            raise
    
    def _handle_bunching_adjustment(self, route_id, trips, reason):
        """Handle bus bunching by adjusting schedules"""
        try:
            # Store original schedule
            original_schedule = []
            for trip in trips:
                original_schedule.append({
                    'trip_id': trip.id,
                    'scheduled_start_time': trip.scheduled_start_time.isoformat(),
                    'scheduled_end_time': trip.scheduled_end_time.isoformat()
                })
            
            # Apply bunching resolution strategy
            adjusted_schedule = self._resolve_bunching(trips)
            
            # Create adjustment record
            adjustment = ScheduleAdjustment(
                route_id=route_id,
                adjustment_type='bunching',
                affected_trips=[trip.id for trip in trips],
                original_schedule=original_schedule,
                adjusted_schedule=adjusted_schedule,
                reason=reason or "Automatic bunching resolution",
                status='pending'
            )
            
            db.session.add(adjustment)
            db.session.commit()
            
            return adjustment
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error handling bunching adjustment: {str(e)}")
            raise
    
    def _handle_delay_adjustment(self, route_id, trips, reason):
        """Handle delays by adjusting subsequent trips"""
        try:
            # Find delayed trips
            delayed_trips = []
            for trip in trips:
                if trip.actual_start_time and trip.actual_start_time > trip.scheduled_start_time:
                    delay_minutes = (trip.actual_start_time - trip.scheduled_start_time).total_seconds() / 60
                    if delay_minutes > 5:  # More than 5 minutes delay
                        delayed_trips.append((trip, delay_minutes))
            
            if not delayed_trips:
                raise ValueError("No significant delays found")
            
            # Store original schedule
            original_schedule = []
            for trip in trips:
                original_schedule.append({
                    'trip_id': trip.id,
                    'scheduled_start_time': trip.scheduled_start_time.isoformat(),
                    'scheduled_end_time': trip.scheduled_end_time.isoformat()
                })
            
            # Apply delay propagation
            adjusted_schedule = self._propagate_delays(trips, delayed_trips)
            
            # Create adjustment record
            adjustment = ScheduleAdjustment(
                route_id=route_id,
                adjustment_type='delay',
                affected_trips=[trip.id for trip in trips],
                original_schedule=original_schedule,
                adjusted_schedule=adjusted_schedule,
                reason=reason or "Automatic delay adjustment",
                status='pending'
            )
            
            db.session.add(adjustment)
            db.session.commit()
            
            return adjustment
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error handling delay adjustment: {str(e)}")
            raise
    
    def _handle_breakdown_adjustment(self, route_id, trips, reason):
        """Handle bus breakdowns by reassigning trips"""
        try:
            # Find trips that need reassignment (simplified logic)
            # In a real system, this would be triggered by breakdown events
            
            # Store original schedule
            original_schedule = []
            for trip in trips:
                original_schedule.append({
                    'trip_id': trip.id,
                    'bus_id': trip.bus_id,
                    'scheduled_start_time': trip.scheduled_start_time.isoformat(),
                    'scheduled_end_time': trip.scheduled_end_time.isoformat()
                })
            
            # Apply breakdown resolution
            adjusted_schedule = self._resolve_breakdown(trips)
            
            # Create adjustment record
            adjustment = ScheduleAdjustment(
                route_id=route_id,
                adjustment_type='breakdown',
                affected_trips=[trip.id for trip in trips],
                original_schedule=original_schedule,
                adjusted_schedule=adjusted_schedule,
                reason=reason or "Automatic breakdown resolution",
                status='pending'
            )
            
            db.session.add(adjustment)
            db.session.commit()
            
            return adjustment
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error handling breakdown adjustment: {str(e)}")
            raise
    
    def _resolve_bunching(self, trips):
        """Resolve bunching by adjusting trip times"""
        adjusted_schedule = []
        
        for i, trip in enumerate(trips):
            if i == 0:
                # Keep first trip as is
                adjusted_schedule.append({
                    'trip_id': trip.id,
                    'scheduled_start_time': trip.scheduled_start_time.isoformat(),
                    'scheduled_end_time': trip.scheduled_end_time.isoformat(),
                    'adjustment_minutes': 0
                })
            else:
                # Ensure minimum headway from previous trip
                prev_trip = trips[i-1]
                min_start_time = prev_trip.scheduled_start_time + timedelta(minutes=self.min_headway_minutes)
                
                if trip.scheduled_start_time < min_start_time:
                    # Adjust this trip to maintain headway
                    adjustment_minutes = (min_start_time - trip.scheduled_start_time).total_seconds() / 60
                    new_start_time = min_start_time
                    new_end_time = trip.scheduled_end_time + timedelta(minutes=adjustment_minutes)
                    
                    adjusted_schedule.append({
                        'trip_id': trip.id,
                        'scheduled_start_time': new_start_time.isoformat(),
                        'scheduled_end_time': new_end_time.isoformat(),
                        'adjustment_minutes': adjustment_minutes
                    })
                else:
                    adjusted_schedule.append({
                        'trip_id': trip.id,
                        'scheduled_start_time': trip.scheduled_start_time.isoformat(),
                        'scheduled_end_time': trip.scheduled_end_time.isoformat(),
                        'adjustment_minutes': 0
                    })
        
        return adjusted_schedule
    
    def _propagate_delays(self, trips, delayed_trips):
        """Propagate delays to subsequent trips"""
        adjusted_schedule = []
        delay_propagation = {}
        
        # Calculate delay propagation
        for trip, delay_minutes in delayed_trips:
            delay_propagation[trip.id] = delay_minutes
        
        for trip in trips:
            if trip.id in delay_propagation:
                # This trip is delayed
                delay = delay_propagation[trip.id]
                new_start_time = trip.scheduled_start_time + timedelta(minutes=delay)
                new_end_time = trip.scheduled_end_time + timedelta(minutes=delay)
                
                adjusted_schedule.append({
                    'trip_id': trip.id,
                    'scheduled_start_time': new_start_time.isoformat(),
                    'scheduled_end_time': new_end_time.isoformat(),
                    'adjustment_minutes': delay
                })
            else:
                # Check if this trip should be delayed due to previous delays
                prev_trips = [t for t in trips if t.scheduled_start_time < trip.scheduled_start_time]
                max_prev_delay = 0
                
                for prev_trip in prev_trips:
                    if prev_trip.id in delay_propagation:
                        max_prev_delay = max(max_prev_delay, delay_propagation[prev_trip.id])
                
                if max_prev_delay > 0:
                    # Apply some delay to maintain service quality
                    propagated_delay = min(max_prev_delay * 0.5, 10)  # Max 10 minutes
                    new_start_time = trip.scheduled_start_time + timedelta(minutes=propagated_delay)
                    new_end_time = trip.scheduled_end_time + timedelta(minutes=propagated_delay)
                    
                    adjusted_schedule.append({
                        'trip_id': trip.id,
                        'scheduled_start_time': new_start_time.isoformat(),
                        'scheduled_end_time': new_end_time.isoformat(),
                        'adjustment_minutes': propagated_delay
                    })
                else:
                    adjusted_schedule.append({
                        'trip_id': trip.id,
                        'scheduled_start_time': trip.scheduled_start_time.isoformat(),
                        'scheduled_end_time': trip.scheduled_end_time.isoformat(),
                        'adjustment_minutes': 0
                    })
        
        return adjusted_schedule
    
    def _resolve_breakdown(self, trips):
        """Resolve breakdowns by reassigning buses"""
        adjusted_schedule = []
        
        # Get available buses for the route
        route_id = trips[0].route_id
        available_buses = Bus.query.filter(
            and_(
                Bus.route_id == route_id,
                Bus.is_active == True
            )
        ).all()
        
        # Simple reassignment logic (in practice, this would be more sophisticated)
        for trip in trips:
            # For now, just keep the same schedule but mark for reassignment
            adjusted_schedule.append({
                'trip_id': trip.id,
                'bus_id': trip.bus_id,
                'scheduled_start_time': trip.scheduled_start_time.isoformat(),
                'scheduled_end_time': trip.scheduled_end_time.isoformat(),
                'reassignment_required': True
            })
        
        return adjusted_schedule
    
    def apply_schedule_adjustment(self, adjustment_id):
        """Apply a schedule adjustment to the database"""
        try:
            adjustment = ScheduleAdjustment.query.get(adjustment_id)
            if not adjustment:
                raise ValueError(f"Adjustment {adjustment_id} not found")
            
            if adjustment.status != 'pending':
                raise ValueError(f"Adjustment {adjustment_id} is not pending")
            
            # Apply the adjusted schedule
            for trip_data in adjustment.adjusted_schedule:
                trip = Trip.query.get(trip_data['trip_id'])
                if trip:
                    trip.scheduled_start_time = datetime.fromisoformat(trip_data['scheduled_start_time'])
                    trip.scheduled_end_time = datetime.fromisoformat(trip_data['scheduled_end_time'])
            
            # Update adjustment status
            adjustment.status = 'applied'
            adjustment.applied_at = datetime.utcnow()
            
            db.session.commit()
            
            logger.info(f"Applied schedule adjustment {adjustment_id}")
            return True
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error applying schedule adjustment {adjustment_id}: {str(e)}")
            raise
    
    def _get_latest_gps_event(self, trip_id):
        """Get the latest GPS event for a trip"""
        return Event.query.filter(
            and_(
                Event.trip_id == trip_id,
                Event.event_type == 'gps_update'
            )
        ).order_by(desc(Event.timestamp)).first()
    
    def _calculate_bunching_severity(self, time_diff_minutes, distance_km):
        """Calculate bunching severity score"""
        if time_diff_minutes <= 1 and distance_km <= 0.5:
            return 'high'
        elif time_diff_minutes <= 2 and distance_km <= 1.0:
            return 'medium'
        else:
            return 'low'
    
    def get_route_headway_analysis(self, route_id, hours=24):
        """Analyze headway consistency for a route"""
        try:
            since_time = datetime.utcnow() - timedelta(hours=hours)
            
            trips = Trip.query.filter(
                and_(
                    Trip.route_id == route_id,
                    Trip.scheduled_start_time >= since_time,
                    Trip.status.in_(['completed', 'in_progress'])
                )
            ).order_by(Trip.scheduled_start_time).all()
            
            if len(trips) < 2:
                return {'error': 'Insufficient trip data'}
            
            headways = []
            for i in range(len(trips) - 1):
                current_trip = trips[i]
                next_trip = trips[i + 1]
                
                headway_minutes = (next_trip.scheduled_start_time - current_trip.scheduled_start_time).total_seconds() / 60
                headways.append(headway_minutes)
            
            if not headways:
                return {'error': 'No headway data available'}
            
            avg_headway = sum(headways) / len(headways)
            min_headway = min(headways)
            max_headway = max(headways)
            
            # Calculate headway consistency (lower is better)
            variance = sum((h - avg_headway) ** 2 for h in headways) / len(headways)
            consistency_score = 1 / (1 + variance)  # Normalized score
            
            return {
                'route_id': route_id,
                'period_hours': hours,
                'total_trips': len(trips),
                'average_headway_minutes': round(avg_headway, 2),
                'min_headway_minutes': round(min_headway, 2),
                'max_headway_minutes': round(max_headway, 2),
                'consistency_score': round(consistency_score, 3),
                'headways': [round(h, 2) for h in headways]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing headway for route {route_id}: {str(e)}")
            return {'error': str(e)}
