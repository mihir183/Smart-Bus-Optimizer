import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import and_, desc, func
from .models import db, Trip, Event, RouteStop, Prediction, Route, Bus
import os
import logging

# Try to import sklearn, but make it optional for basic functionality
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML features will be limited.")

logger = logging.getLogger(__name__)

class Predictor:
    """ML-based arrival time prediction system"""
    
    def __init__(self, model_path='models/'):
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_version = '1.0'
        
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
        
        # Load existing models if available
        self._load_models()
    
    def train_models(self, route_id=None, days_back=30):
        """
        Train prediction models for arrival time
        
        Args:
            route_id (int): Specific route to train for, or None for all routes
            days_back (int): Number of days of historical data to use
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Cannot train models.")
            return False
            
        try:
            logger.info(f"Starting model training for route {route_id or 'all'}")
            
            # Get training data
            training_data = self._prepare_training_data(route_id, days_back)
            
            if training_data.empty:
                logger.warning("No training data available")
                return False
            
            # Prepare features and target
            X, y = self._prepare_features_target(training_data)
            
            if X.empty or len(y) == 0:
                logger.warning("No valid features or targets found")
                return False
            
            # Train models for each route
            routes = training_data['route_id'].unique()
            
            for route in routes:
                route_data = training_data[training_data['route_id'] == route]
                route_X, route_y = self._prepare_features_target(route_data)
                
                if len(route_X) < 10:  # Need minimum data points
                    logger.warning(f"Insufficient data for route {route}")
                    continue
                
                # Train models
                self._train_route_models(route, route_X, route_y)
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
    
    def predict_arrival_time(self, trip, stop):
        """
        Predict arrival time for a specific trip at a specific stop
        
        Args:
            trip (Trip): Trip object
            stop (RouteStop): Stop object
            
        Returns:
            Prediction: Prediction object or None
        """
        if not SKLEARN_AVAILABLE:
            # Fallback to simple prediction without ML
            logger.warning("scikit-learn not available. Using simple prediction.")
            return self._simple_prediction(trip, stop)
            
        try:
            # Check if we have a model for this route
            if trip.route_id not in self.models:
                logger.warning(f"No model available for route {trip.route_id}")
                return self._simple_prediction(trip, stop)
            
            # Get current trip state
            features = self._extract_trip_features(trip, stop)
            
            if features is None:
                return None
            
            # Make prediction
            model = self.models[trip.route_id]['arrival_time']
            scaler = self.scalers[trip.route_id]['arrival_time']
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Predict
            prediction_minutes = model.predict(features_scaled)[0]
            predicted_time = datetime.utcnow() + timedelta(minutes=prediction_minutes)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(trip.route_id, features)
            
            # Create prediction record
            prediction = Prediction(
                trip_id=trip.id,
                stop_id=stop.id,
                predicted_arrival_time=predicted_time,
                confidence_score=confidence,
                model_version=self.model_version,
                features_used=features
            )
            
            db.session.add(prediction)
            db.session.commit()
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            db.session.rollback()
            return self._simple_prediction(trip, stop)
    
    def _simple_prediction(self, trip, stop):
        """Simple prediction without ML when sklearn is not available"""
        try:
            # Simple prediction based on scheduled time and stop sequence
            if stop.estimated_time_from_start:
                predicted_time = trip.scheduled_start_time + timedelta(minutes=stop.estimated_time_from_start)
            else:
                # Fallback: estimate based on route duration and stop position
                route_duration = trip.route.estimated_duration or 60
                stop_progress = stop.sequence_order / 10.0  # Assume 10 stops average
                estimated_minutes = route_duration * stop_progress
                predicted_time = trip.scheduled_start_time + timedelta(minutes=estimated_minutes)
            
            # Create prediction record
            prediction = Prediction(
                trip_id=trip.id,
                stop_id=stop.id,
                predicted_arrival_time=predicted_time,
                confidence_score=0.5,  # Lower confidence for simple prediction
                model_version='simple-1.0',
                features_used={'method': 'simple', 'stop_sequence': stop.sequence_order}
            )
            
            db.session.add(prediction)
            db.session.commit()
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in simple prediction: {str(e)}")
            db.session.rollback()
            return None
    
    def _prepare_training_data(self, route_id, days_back):
        """Prepare training data from historical trips and events"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Get completed trips
            query = Trip.query.filter(
                and_(
                    Trip.status == 'completed',
                    Trip.actual_start_time.isnot(None),
                    Trip.actual_end_time.isnot(None),
                    Trip.scheduled_start_time >= since_date
                )
            )
            
            if route_id:
                query = query.filter_by(route_id=route_id)
            
            trips = query.all()
            
            if not trips:
                return pd.DataFrame()
            
            # Collect training data
            training_data = []
            
            for trip in trips:
                # Get events for this trip
                events = Event.query.filter(
                    and_(
                        Event.trip_id == trip.id,
                        Event.event_type.in_(['gps_update', 'occupancy_update'])
                    )
                ).order_by(Event.timestamp).all()
                
                if not events:
                    continue
                
                # Get route stops
                stops = RouteStop.query.filter_by(route_id=trip.route_id).order_by(RouteStop.sequence_order).all()
                
                # Calculate actual arrival times at stops
                for stop in stops:
                    arrival_time = self._find_stop_arrival_time(trip, stop, events)
                    if arrival_time:
                        # Extract features
                        features = self._extract_historical_features(trip, stop, events, arrival_time)
                        if features:
                            features['actual_arrival_time'] = arrival_time
                            features['trip_id'] = trip.id
                            features['route_id'] = trip.route_id
                            features['stop_id'] = stop.id
                            training_data.append(features)
            
            return pd.DataFrame(training_data)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return pd.DataFrame()
    
    def _find_stop_arrival_time(self, trip, stop, events):
        """Find when the bus actually arrived at a specific stop"""
        try:
            # Find GPS events near the stop
            stop_events = []
            for event in events:
                if event.event_type == 'gps_update' and event.latitude and event.longitude:
                    distance = self._calculate_distance(
                        event.latitude, event.longitude,
                        stop.latitude, stop.longitude
                    )
                    if distance <= 0.1:  # Within 100 meters
                        stop_events.append((event.timestamp, distance))
            
            if not stop_events:
                return None
            
            # Return the timestamp of the closest event
            stop_events.sort(key=lambda x: x[1])  # Sort by distance
            return stop_events[0][0]
            
        except Exception as e:
            logger.error(f"Error finding stop arrival time: {str(e)}")
            return None
    
    def _extract_historical_features(self, trip, stop, events, arrival_time):
        """Extract features from historical trip data"""
        try:
            features = {}
            
            # Basic trip features
            features['route_id'] = trip.route_id
            features['bus_id'] = trip.bus_id
            features['stop_sequence'] = stop.sequence_order
            features['scheduled_start_time'] = trip.scheduled_start_time.hour
            features['day_of_week'] = trip.scheduled_start_time.weekday()
            
            # Calculate delay at start
            if trip.actual_start_time:
                delay_minutes = (trip.actual_start_time - trip.scheduled_start_time).total_seconds() / 60
                features['start_delay_minutes'] = delay_minutes
            else:
                features['start_delay_minutes'] = 0
            
            # Get events before arrival at this stop
            relevant_events = [e for e in events if e.timestamp <= arrival_time]
            
            if relevant_events:
                # Average speed
                speeds = [e.speed for e in relevant_events if e.speed and e.speed > 0]
                features['avg_speed_kmh'] = np.mean(speeds) if speeds else 25
                
                # Average occupancy
                occupancies = [e.occupancy_percentage for e in relevant_events if e.occupancy_percentage is not None]
                features['avg_occupancy_percent'] = np.mean(occupancies) if occupancies else 50
                
                # Recent delay events
                delay_events = [e for e in relevant_events if e.event_type == 'delay']
                features['total_delay_minutes'] = sum(e.delay_minutes or 0 for e in delay_events)
            else:
                features['avg_speed_kmh'] = 25
                features['avg_occupancy_percent'] = 50
                features['total_delay_minutes'] = 0
            
            # Weather and traffic features (simplified)
            features['is_weekend'] = 1 if trip.scheduled_start_time.weekday() >= 5 else 0
            features['is_peak_hour'] = 1 if 7 <= trip.scheduled_start_time.hour <= 9 or 17 <= trip.scheduled_start_time.hour <= 19 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting historical features: {str(e)}")
            return None
    
    def _extract_trip_features(self, trip, stop):
        """Extract current features for prediction"""
        try:
            features = {}
            
            # Basic trip features
            features['route_id'] = trip.route_id
            features['bus_id'] = trip.bus_id
            features['stop_sequence'] = stop.sequence_order
            features['scheduled_start_time'] = trip.scheduled_start_time.hour
            features['day_of_week'] = trip.scheduled_start_time.weekday()
            
            # Current delay
            if trip.actual_start_time:
                delay_minutes = (trip.actual_start_time - trip.scheduled_start_time).total_seconds() / 60
                features['start_delay_minutes'] = delay_minutes
            else:
                features['start_delay_minutes'] = 0
            
            # Get recent events
            recent_events = Event.query.filter(
                and_(
                    Event.trip_id == trip.id,
                    Event.timestamp >= datetime.utcnow() - timedelta(minutes=30)
                )
            ).order_by(desc(Event.timestamp)).limit(10).all()
            
            if recent_events:
                # Current speed
                gps_events = [e for e in recent_events if e.event_type == 'gps_update' and e.speed]
                features['avg_speed_kmh'] = np.mean([e.speed for e in gps_events]) if gps_events else 25
                
                # Current occupancy
                occupancy_events = [e for e in recent_events if e.event_type == 'occupancy_update' and e.occupancy_percentage]
                features['avg_occupancy_percent'] = np.mean([e.occupancy_percentage for e in occupancy_events]) if occupancy_events else 50
                
                # Recent delays
                delay_events = [e for e in recent_events if e.event_type == 'delay']
                features['total_delay_minutes'] = sum(e.delay_minutes or 0 for e in delay_events)
            else:
                features['avg_speed_kmh'] = 25
                features['avg_occupancy_percent'] = 50
                features['total_delay_minutes'] = 0
            
            # Time-based features
            features['is_weekend'] = 1 if trip.scheduled_start_time.weekday() >= 5 else 0
            features['is_peak_hour'] = 1 if 7 <= trip.scheduled_start_time.hour <= 9 or 17 <= trip.scheduled_start_time.hour <= 19 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting trip features: {str(e)}")
            return None
    
    def _prepare_features_target(self, data):
        """Prepare features and target for training"""
        try:
            # Select feature columns
            feature_columns = [
                'route_id', 'bus_id', 'stop_sequence', 'scheduled_start_time',
                'day_of_week', 'start_delay_minutes', 'avg_speed_kmh',
                'avg_occupancy_percent', 'total_delay_minutes', 'is_weekend', 'is_peak_hour'
            ]
            
            X = data[feature_columns].copy()
            
            # Calculate target (minutes until arrival)
            now = datetime.utcnow()
            y = []
            for _, row in data.iterrows():
                arrival_time = row['actual_arrival_time']
                minutes_until_arrival = (arrival_time - now).total_seconds() / 60
                y.append(max(0, minutes_until_arrival))  # Ensure non-negative
            
            return X, np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing features and target: {str(e)}")
            return pd.DataFrame(), np.array([])
    
    def _train_route_models(self, route_id, X, y):
        """Train models for a specific route"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize models
            models = {
                'arrival_time': RandomForestRegressor(n_estimators=100, random_state=42),
                'backup': GradientBoostingRegressor(random_state=42)
            }
            
            # Train models
            for name, model in models.items():
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"Route {route_id} - {name} model: MAE={mae:.2f}, MSE={mse:.2f}, R²={r2:.3f}")
            
            # Create scaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            
            # Store models and scaler
            self.models[route_id] = models
            self.scalers[route_id] = {'arrival_time': scaler}
            
            # Save models
            self._save_models(route_id)
            
        except Exception as e:
            logger.error(f"Error training models for route {route_id}: {str(e)}")
    
    def _calculate_confidence(self, route_id, features):
        """Calculate confidence score for prediction"""
        try:
            # Simple confidence calculation based on feature quality
            confidence = 0.8  # Base confidence
            
            # Adjust based on data availability
            if features.get('avg_speed_kmh', 0) > 0:
                confidence += 0.1
            
            if features.get('avg_occupancy_percent', 0) > 0:
                confidence += 0.1
            
            # Adjust based on delay
            if features.get('total_delay_minutes', 0) > 10:
                confidence -= 0.2
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _save_models(self, route_id):
        """Save trained models to disk"""
        try:
            route_path = os.path.join(self.model_path, f'route_{route_id}')
            os.makedirs(route_path, exist_ok=True)
            
            # Save models
            for name, model in self.models[route_id].items():
                model_file = os.path.join(route_path, f'{name}_model.pkl')
                joblib.dump(model, model_file)
            
            # Save scaler
            scaler_file = os.path.join(route_path, 'scaler.pkl')
            joblib.dump(self.scalers[route_id]['arrival_time'], scaler_file)
            
        except Exception as e:
            logger.error(f"Error saving models for route {route_id}: {str(e)}")
    
    def _load_models(self):
        """Load existing models from disk"""
        try:
            if not os.path.exists(self.model_path):
                return
            
            for route_dir in os.listdir(self.model_path):
                if route_dir.startswith('route_'):
                    route_id = int(route_dir.split('_')[1])
                    route_path = os.path.join(self.model_path, route_dir)
                    
                    # Load models
                    models = {}
                    for model_file in os.listdir(route_path):
                        if model_file.endswith('_model.pkl'):
                            name = model_file.replace('_model.pkl', '')
                            model_path = os.path.join(route_path, model_file)
                            models[name] = joblib.load(model_path)
                    
                    # Load scaler
                    scaler_file = os.path.join(route_path, 'scaler.pkl')
                    if os.path.exists(scaler_file):
                        scaler = joblib.load(scaler_file)
                        self.models[route_id] = models
                        self.scalers[route_id] = {'arrival_time': scaler}
                        
                        logger.info(f"Loaded models for route {route_id}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
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
    
    def get_model_performance(self, route_id):
        """Get model performance metrics"""
        try:
            if route_id not in self.models:
                return None
            
            # Get recent predictions and actual arrivals
            since_date = datetime.utcnow() - timedelta(days=7)
            
            predictions = Prediction.query.filter(
                and_(
                    Prediction.trip_id.in_(
                        db.session.query(Trip.id).filter(
                            Trip.route_id == route_id,
                            Trip.scheduled_start_time >= since_date
                        )
                    ),
                    Prediction.created_at >= since_date
                )
            ).all()
            
            if not predictions:
                return {'error': 'No recent predictions found'}
            
            # Calculate accuracy metrics
            errors = []
            for pred in predictions:
                # Find actual arrival time
                actual_arrival = self._find_actual_arrival_time(pred.trip_id, pred.stop_id)
                if actual_arrival:
                    error_minutes = abs((pred.predicted_arrival_time - actual_arrival).total_seconds() / 60)
                    errors.append(error_minutes)
            
            if not errors:
                return {'error': 'No actual arrival times found'}
            
            return {
                'route_id': route_id,
                'total_predictions': len(predictions),
                'predictions_with_actuals': len(errors),
                'mean_absolute_error_minutes': round(np.mean(errors), 2),
                'median_absolute_error_minutes': round(np.median(errors), 2),
                'accuracy_within_2_minutes': round(sum(1 for e in errors if e <= 2) / len(errors) * 100, 1),
                'accuracy_within_5_minutes': round(sum(1 for e in errors if e <= 5) / len(errors) * 100, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return {'error': str(e)}
    
    def _find_actual_arrival_time(self, trip_id, stop_id):
        """Find actual arrival time for a trip at a stop"""
        try:
            # Get the stop
            stop = RouteStop.query.get(stop_id)
            if not stop:
                return None
            
            # Get GPS events for the trip near the stop
            events = Event.query.filter(
                and_(
                    Event.trip_id == trip_id,
                    Event.event_type == 'gps_update',
                    Event.latitude.isnot(None),
                    Event.longitude.isnot(None)
                )
            ).order_by(Event.timestamp).all()
            
            # Find closest event to stop
            min_distance = float('inf')
            closest_time = None
            
            for event in events:
                distance = self._calculate_distance(
                    event.latitude, event.longitude,
                    stop.latitude, stop.longitude
                )
                if distance < min_distance and distance <= 0.1:  # Within 100 meters
                    min_distance = distance
                    closest_time = event.timestamp
            
            return closest_time
            
        except Exception as e:
            logger.error(f"Error finding actual arrival time: {str(e)}")
            return None
