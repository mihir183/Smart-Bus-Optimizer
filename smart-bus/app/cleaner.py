import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import and_, desc, func
from .models import db, Trip, Event, RouteStop, Route, Bus
import logging
import os

logger = logging.getLogger(__name__)

class DataCleaner:
    """Data cleaning pipeline for bus system data"""
    
    def __init__(self, raw_data_path='data/raw/', clean_data_path='data/clean/'):
        self.raw_data_path = raw_data_path
        self.clean_data_path = clean_data_path
        
        # Ensure directories exist
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(clean_data_path, exist_ok=True)
        
        # Data quality thresholds
        self.gps_accuracy_threshold = 100  # meters
        self.speed_threshold_kmh = 80  # Maximum reasonable bus speed
        self.occupancy_threshold = 120  # Maximum occupancy percentage
        
    def clean_gps_data(self, file_path=None, days_back=7):
        """
        Clean GPS data from CSV file or database
        
        Args:
            file_path (str): Path to GPS CSV file, or None to use database
            days_back (int): Number of days to clean from database
        """
        try:
            if file_path:
                # Clean from CSV file
                df = pd.read_csv(file_path)
                cleaned_df = self._clean_gps_dataframe(df)
                
                # Save cleaned data
                output_path = os.path.join(self.clean_data_path, 'gps_logs_clean.csv')
                cleaned_df.to_csv(output_path, index=False)
                
                logger.info(f"Cleaned GPS data saved to {output_path}")
                return cleaned_df
            else:
                # Clean from database
                return self._clean_gps_database(days_back)
                
        except Exception as e:
            logger.error(f"Error cleaning GPS data: {str(e)}")
            return None
    
    def clean_occupancy_data(self, file_path=None, days_back=7):
        """
        Clean occupancy data from CSV file or database
        
        Args:
            file_path (str): Path to occupancy CSV file, or None to use database
            days_back (int): Number of days to clean from database
        """
        try:
            if file_path:
                # Clean from CSV file
                df = pd.read_csv(file_path)
                cleaned_df = self._clean_occupancy_dataframe(df)
                
                # Save cleaned data
                output_path = os.path.join(self.clean_data_path, 'occupancy_clean.csv')
                cleaned_df.to_csv(output_path, index=False)
                
                logger.info(f"Cleaned occupancy data saved to {output_path}")
                return cleaned_df
            else:
                # Clean from database
                return self._clean_occupancy_database(days_back)
                
        except Exception as e:
            logger.error(f"Error cleaning occupancy data: {str(e)}")
            return None
    
    def clean_trip_data(self, file_path=None, days_back=7):
        """
        Clean trip data from CSV file or database
        
        Args:
            file_path (str): Path to trip CSV file, or None to use database
            days_back (int): Number of days to clean from database
        """
        try:
            if file_path:
                # Clean from CSV file
                df = pd.read_csv(file_path)
                cleaned_df = self._clean_trip_dataframe(df)
                
                # Save cleaned data
                output_path = os.path.join(self.clean_data_path, 'trips_clean.csv')
                cleaned_df.to_csv(output_path, index=False)
                
                logger.info(f"Cleaned trip data saved to {output_path}")
                return cleaned_df
            else:
                # Clean from database
                return self._clean_trip_database(days_back)
                
        except Exception as e:
            logger.error(f"Error cleaning trip data: {str(e)}")
            return None
    
    def _clean_gps_dataframe(self, df):
        """Clean GPS data from DataFrame"""
        try:
            # Make a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Convert timestamp column
            if 'timestamp' in cleaned_df.columns:
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'], errors='coerce')
            
            # Remove rows with invalid timestamps
            cleaned_df = cleaned_df.dropna(subset=['timestamp'])
            
            # Remove rows with invalid coordinates
            cleaned_df = cleaned_df.dropna(subset=['latitude', 'longitude'])
            
            # Filter out invalid coordinates
            cleaned_df = cleaned_df[
                (cleaned_df['latitude'] >= -90) & (cleaned_df['latitude'] <= 90) &
                (cleaned_df['longitude'] >= -180) & (cleaned_df['longitude'] <= 180)
            ]
            
            # Clean speed data
            if 'speed' in cleaned_df.columns:
                # Remove negative speeds and unreasonably high speeds
                cleaned_df = cleaned_df[
                    (cleaned_df['speed'] >= 0) & (cleaned_df['speed'] <= self.speed_threshold_kmh)
                ]
                
                # Fill missing speeds with median
                cleaned_df['speed'] = cleaned_df['speed'].fillna(cleaned_df['speed'].median())
            
            # Clean heading data
            if 'heading' in cleaned_df.columns:
                # Normalize heading to 0-360 range
                cleaned_df['heading'] = cleaned_df['heading'] % 360
                cleaned_df['heading'] = cleaned_df['heading'].fillna(0)
            
            # Remove duplicate entries
            cleaned_df = cleaned_df.drop_duplicates(subset=['trip_id', 'timestamp'], keep='first')
            
            # Sort by timestamp
            cleaned_df = cleaned_df.sort_values('timestamp')
            
            # Remove outliers using IQR method for coordinates
            for coord in ['latitude', 'longitude']:
                if coord in cleaned_df.columns:
                    Q1 = cleaned_df[coord].quantile(0.25)
                    Q3 = cleaned_df[coord].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    cleaned_df = cleaned_df[
                        (cleaned_df[coord] >= lower_bound) & (cleaned_df[coord] <= upper_bound)
                    ]
            
            logger.info(f"GPS data cleaning completed. Removed {len(df) - len(cleaned_df)} invalid records")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning GPS DataFrame: {str(e)}")
            return df
    
    def _clean_occupancy_dataframe(self, df):
        """Clean occupancy data from DataFrame"""
        try:
            cleaned_df = df.copy()
            
            # Convert timestamp column
            if 'timestamp' in cleaned_df.columns:
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'], errors='coerce')
            
            # Remove rows with invalid timestamps
            cleaned_df = cleaned_df.dropna(subset=['timestamp'])
            
            # Clean occupancy count
            if 'occupancy_count' in cleaned_df.columns:
                # Remove negative occupancy
                cleaned_df = cleaned_df[cleaned_df['occupancy_count'] >= 0]
                
                # Remove unreasonably high occupancy (more than 120% of capacity)
                if 'capacity' in cleaned_df.columns:
                    cleaned_df = cleaned_df[cleaned_df['occupancy_count'] <= cleaned_df['capacity'] * 1.2]
            
            # Clean occupancy percentage
            if 'occupancy_percentage' in cleaned_df.columns:
                # Remove negative percentages
                cleaned_df = cleaned_df[cleaned_df['occupancy_percentage'] >= 0]
                
                # Remove percentages over 120%
                cleaned_df = cleaned_df[cleaned_df['occupancy_percentage'] <= self.occupancy_threshold]
            
            # Remove duplicate entries
            cleaned_df = cleaned_df.drop_duplicates(subset=['trip_id', 'timestamp'], keep='first')
            
            # Sort by timestamp
            cleaned_df = cleaned_df.sort_values('timestamp')
            
            logger.info(f"Occupancy data cleaning completed. Removed {len(df) - len(cleaned_df)} invalid records")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning occupancy DataFrame: {str(e)}")
            return df
    
    def _clean_trip_dataframe(self, df):
        """Clean trip data from DataFrame"""
        try:
            cleaned_df = df.copy()
            
            # Convert timestamp columns
            timestamp_columns = ['scheduled_start_time', 'actual_start_time', 'scheduled_end_time', 'actual_end_time']
            for col in timestamp_columns:
                if col in cleaned_df.columns:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            
            # Remove rows with invalid scheduled times
            cleaned_df = cleaned_df.dropna(subset=['scheduled_start_time', 'scheduled_end_time'])
            
            # Validate trip duration (should be reasonable)
            cleaned_df['duration_minutes'] = (
                cleaned_df['scheduled_end_time'] - cleaned_df['scheduled_start_time']
            ).dt.total_seconds() / 60
            
            # Remove trips with unreasonable duration (less than 5 minutes or more than 3 hours)
            cleaned_df = cleaned_df[
                (cleaned_df['duration_minutes'] >= 5) & (cleaned_df['duration_minutes'] <= 180)
            ]
            
            # Clean actual times
            if 'actual_start_time' in cleaned_df.columns:
                # Remove actual start times that are too far from scheduled
                time_diff = (cleaned_df['actual_start_time'] - cleaned_df['scheduled_start_time']).dt.total_seconds() / 60
                cleaned_df = cleaned_df[abs(time_diff) <= 60]  # Within 1 hour
            
            # Clean status values
            if 'status' in cleaned_df.columns:
                valid_statuses = ['scheduled', 'in_progress', 'completed', 'cancelled']
                cleaned_df = cleaned_df[cleaned_df['status'].isin(valid_statuses)]
            
            # Remove duplicate entries
            cleaned_df = cleaned_df.drop_duplicates(subset=['trip_id'], keep='first')
            
            # Sort by scheduled start time
            cleaned_df = cleaned_df.sort_values('scheduled_start_time')
            
            logger.info(f"Trip data cleaning completed. Removed {len(df) - len(cleaned_df)} invalid records")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning trip DataFrame: {str(e)}")
            return df
    
    def _clean_gps_database(self, days_back):
        """Clean GPS data from database"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Get GPS events
            events = Event.query.filter(
                and_(
                    Event.event_type == 'gps_update',
                    Event.timestamp >= since_date
                )
            ).all()
            
            if not events:
                logger.warning("No GPS events found in database")
                return None
            
            # Convert to DataFrame
            data = []
            for event in events:
                data.append({
                    'id': event.id,
                    'trip_id': event.trip_id,
                    'bus_id': event.bus_id,
                    'timestamp': event.timestamp,
                    'latitude': event.latitude,
                    'longitude': event.longitude,
                    'speed': event.speed,
                    'heading': event.heading
                })
            
            df = pd.DataFrame(data)
            cleaned_df = self._clean_gps_dataframe(df)
            
            # Update database with cleaned data
            self._update_gps_events(cleaned_df)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning GPS database: {str(e)}")
            return None
    
    def _clean_occupancy_database(self, days_back):
        """Clean occupancy data from database"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Get occupancy events
            events = Event.query.filter(
                and_(
                    Event.event_type == 'occupancy_update',
                    Event.timestamp >= since_date
                )
            ).all()
            
            if not events:
                logger.warning("No occupancy events found in database")
                return None
            
            # Convert to DataFrame
            data = []
            for event in events:
                data.append({
                    'id': event.id,
                    'trip_id': event.trip_id,
                    'bus_id': event.bus_id,
                    'timestamp': event.timestamp,
                    'occupancy_count': event.occupancy_count,
                    'occupancy_percentage': event.occupancy_percentage
                })
            
            df = pd.DataFrame(data)
            cleaned_df = self._clean_occupancy_dataframe(df)
            
            # Update database with cleaned data
            self._update_occupancy_events(cleaned_df)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning occupancy database: {str(e)}")
            return None
    
    def _clean_trip_database(self, days_back):
        """Clean trip data from database"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Get trips
            trips = Trip.query.filter(
                Trip.scheduled_start_time >= since_date
            ).all()
            
            if not trips:
                logger.warning("No trips found in database")
                return None
            
            # Convert to DataFrame
            data = []
            for trip in trips:
                data.append({
                    'id': trip.id,
                    'route_id': trip.route_id,
                    'bus_id': trip.bus_id,
                    'scheduled_start_time': trip.scheduled_start_time,
                    'actual_start_time': trip.actual_start_time,
                    'scheduled_end_time': trip.scheduled_end_time,
                    'actual_end_time': trip.actual_end_time,
                    'status': trip.status
                })
            
            df = pd.DataFrame(data)
            cleaned_df = self._clean_trip_dataframe(df)
            
            # Update database with cleaned data
            self._update_trips(cleaned_df)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning trip database: {str(e)}")
            return None
    
    def _update_gps_events(self, cleaned_df):
        """Update GPS events in database with cleaned data"""
        try:
            for _, row in cleaned_df.iterrows():
                event = Event.query.get(row['id'])
                if event:
                    event.latitude = row['latitude']
                    event.longitude = row['longitude']
                    event.speed = row['speed']
                    event.heading = row['heading']
            
            db.session.commit()
            logger.info(f"Updated {len(cleaned_df)} GPS events in database")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating GPS events: {str(e)}")
    
    def _update_occupancy_events(self, cleaned_df):
        """Update occupancy events in database with cleaned data"""
        try:
            for _, row in cleaned_df.iterrows():
                event = Event.query.get(row['id'])
                if event:
                    event.occupancy_count = row['occupancy_count']
                    event.occupancy_percentage = row['occupancy_percentage']
            
            db.session.commit()
            logger.info(f"Updated {len(cleaned_df)} occupancy events in database")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating occupancy events: {str(e)}")
    
    def _update_trips(self, cleaned_df):
        """Update trips in database with cleaned data"""
        try:
            for _, row in cleaned_df.iterrows():
                trip = Trip.query.get(row['id'])
                if trip:
                    trip.actual_start_time = row['actual_start_time']
                    trip.actual_end_time = row['actual_end_time']
                    trip.status = row['status']
            
            db.session.commit()
            logger.info(f"Updated {len(cleaned_df)} trips in database")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating trips: {str(e)}")
    
    def detect_anomalies(self, data_type='gps', days_back=7):
        """
        Detect anomalies in the data
        
        Args:
            data_type (str): Type of data to analyze ('gps', 'occupancy', 'trip')
            days_back (int): Number of days to analyze
        """
        try:
            since_date = datetime.utcnow() - timedelta(days=days_back)
            anomalies = []
            
            if data_type == 'gps':
                anomalies = self._detect_gps_anomalies(since_date)
            elif data_type == 'occupancy':
                anomalies = self._detect_occupancy_anomalies(since_date)
            elif data_type == 'trip':
                anomalies = self._detect_trip_anomalies(since_date)
            
            logger.info(f"Detected {len(anomalies)} anomalies in {data_type} data")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def _detect_gps_anomalies(self, since_date):
        """Detect GPS anomalies"""
        anomalies = []
        
        # Get GPS events
        events = Event.query.filter(
            and_(
                Event.event_type == 'gps_update',
                Event.timestamp >= since_date
            )
        ).order_by(Event.trip_id, Event.timestamp).all()
        
        # Group by trip
        trip_events = {}
        for event in events:
            if event.trip_id not in trip_events:
                trip_events[event.trip_id] = []
            trip_events[event.trip_id].append(event)
        
        # Detect anomalies in each trip
        for trip_id, trip_event_list in trip_events.items():
            for i in range(1, len(trip_event_list)):
                prev_event = trip_event_list[i-1]
                curr_event = trip_event_list[i]
                
                # Calculate distance and time
                if prev_event.latitude and prev_event.longitude and curr_event.latitude and curr_event.longitude:
                    distance = self._calculate_distance(
                        prev_event.latitude, prev_event.longitude,
                        curr_event.latitude, curr_event.longitude
                    )
                    time_diff = (curr_event.timestamp - prev_event.timestamp).total_seconds() / 3600  # hours
                    
                    if time_diff > 0:
                        speed = distance / time_diff  # km/h
                        
                        # Detect speed anomalies
                        if speed > self.speed_threshold_kmh:
                            anomalies.append({
                                'type': 'speed_anomaly',
                                'trip_id': trip_id,
                                'event_id': curr_event.id,
                                'timestamp': curr_event.timestamp,
                                'speed_kmh': speed,
                                'threshold': self.speed_threshold_kmh
                            })
                        
                        # Detect location jumps
                        if distance > 5:  # More than 5 km jump
                            anomalies.append({
                                'type': 'location_jump',
                                'trip_id': trip_id,
                                'event_id': curr_event.id,
                                'timestamp': curr_event.timestamp,
                                'distance_km': distance
                            })
        
        return anomalies
    
    def _detect_occupancy_anomalies(self, since_date):
        """Detect occupancy anomalies"""
        anomalies = []
        
        # Get occupancy events
        events = Event.query.filter(
            and_(
                Event.event_type == 'occupancy_update',
                Event.timestamp >= since_date
            )
        ).order_by(Event.trip_id, Event.timestamp).all()
        
        # Group by trip
        trip_events = {}
        for event in events:
            if event.trip_id not in trip_events:
                trip_events[event.trip_id] = []
            trip_events[event.trip_id].append(event)
        
        # Detect anomalies in each trip
        for trip_id, trip_event_list in trip_events.items():
            for i in range(1, len(trip_event_list)):
                prev_event = trip_event_list[i-1]
                curr_event = trip_event_list[i]
                
                if prev_event.occupancy_count is not None and curr_event.occupancy_count is not None:
                    # Detect sudden occupancy changes
                    occupancy_change = abs(curr_event.occupancy_count - prev_event.occupancy_count)
                    time_diff = (curr_event.timestamp - prev_event.timestamp).total_seconds() / 60  # minutes
                    
                    if time_diff > 0 and occupancy_change / time_diff > 10:  # More than 10 passengers per minute
                        anomalies.append({
                            'type': 'occupancy_spike',
                            'trip_id': trip_id,
                            'event_id': curr_event.id,
                            'timestamp': curr_event.timestamp,
                            'occupancy_change': occupancy_change,
                            'time_diff_minutes': time_diff
                        })
        
        return anomalies
    
    def _detect_trip_anomalies(self, since_date):
        """Detect trip anomalies"""
        anomalies = []
        
        # Get trips
        trips = Trip.query.filter(
            Trip.scheduled_start_time >= since_date
        ).all()
        
        for trip in trips:
            # Detect excessive delays
            if trip.actual_start_time and trip.scheduled_start_time:
                delay = (trip.actual_start_time - trip.scheduled_start_time).total_seconds() / 60
                if delay > 30:  # More than 30 minutes delay
                    anomalies.append({
                        'type': 'excessive_delay',
                        'trip_id': trip.id,
                        'route_id': trip.route_id,
                        'delay_minutes': delay,
                        'scheduled_start': trip.scheduled_start_time,
                        'actual_start': trip.actual_start_time
                    })
            
            # Detect very short or long trips
            if trip.scheduled_start_time and trip.scheduled_end_time:
                duration = (trip.scheduled_end_time - trip.scheduled_start_time).total_seconds() / 60
                if duration < 5 or duration > 180:  # Less than 5 minutes or more than 3 hours
                    anomalies.append({
                        'type': 'unusual_duration',
                        'trip_id': trip.id,
                        'route_id': trip.route_id,
                        'duration_minutes': duration,
                        'scheduled_start': trip.scheduled_start_time,
                        'scheduled_end': trip.scheduled_end_time
                    })
        
        return anomalies
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in kilometers"""
        import math
        
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def generate_data_quality_report(self, days_back=7):
        """Generate a comprehensive data quality report"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days_back)
            
            report = {
                'period': f"Last {days_back} days",
                'generated_at': datetime.utcnow().isoformat(),
                'gps_data': self._analyze_gps_quality(since_date),
                'occupancy_data': self._analyze_occupancy_quality(since_date),
                'trip_data': self._analyze_trip_quality(since_date),
                'anomalies': {
                    'gps': len(self._detect_gps_anomalies(since_date)),
                    'occupancy': len(self._detect_occupancy_anomalies(since_date)),
                    'trip': len(self._detect_trip_anomalies(since_date))
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {str(e)}")
            return None
    
    def _analyze_gps_quality(self, since_date):
        """Analyze GPS data quality"""
        total_events = Event.query.filter(
            and_(
                Event.event_type == 'gps_update',
                Event.timestamp >= since_date
            )
        ).count()
        
        valid_events = Event.query.filter(
            and_(
                Event.event_type == 'gps_update',
                Event.timestamp >= since_date,
                Event.latitude.isnot(None),
                Event.longitude.isnot(None)
            )
        ).count()
        
        return {
            'total_events': total_events,
            'valid_events': valid_events,
            'completeness_percentage': round((valid_events / total_events * 100) if total_events > 0 else 0, 2)
        }
    
    def _analyze_occupancy_quality(self, since_date):
        """Analyze occupancy data quality"""
        total_events = Event.query.filter(
            and_(
                Event.event_type == 'occupancy_update',
                Event.timestamp >= since_date
            )
        ).count()
        
        valid_events = Event.query.filter(
            and_(
                Event.event_type == 'occupancy_update',
                Event.timestamp >= since_date,
                Event.occupancy_count.isnot(None)
            )
        ).count()
        
        return {
            'total_events': total_events,
            'valid_events': valid_events,
            'completeness_percentage': round((valid_events / total_events * 100) if total_events > 0 else 0, 2)
        }
    
    def _analyze_trip_quality(self, since_date):
        """Analyze trip data quality"""
        total_trips = Trip.query.filter(
            Trip.scheduled_start_time >= since_date
        ).count()
        
        completed_trips = Trip.query.filter(
            and_(
                Trip.scheduled_start_time >= since_date,
                Trip.status == 'completed',
                Trip.actual_start_time.isnot(None),
                Trip.actual_end_time.isnot(None)
            )
        ).count()
        
        return {
            'total_trips': total_trips,
            'completed_trips': completed_trips,
            'completion_percentage': round((completed_trips / total_trips * 100) if total_trips > 0 else 0, 2)
        }
