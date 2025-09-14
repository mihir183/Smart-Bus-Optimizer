#!/usr/bin/env python3
"""
Synthetic Data Generator for Smart Bus System

This script generates realistic synthetic data for development and testing purposes.
It creates sample routes, buses, trips, and events that simulate real bus operations.
"""

import os
import sys
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

from app import create_app, db
from app.models import Route, Bus, Trip, Event, RouteStop, Prediction

class SyntheticDataGenerator:
    """Generate synthetic data for the smart bus system"""
    
    def __init__(self):
        self.app = create_app()
        self.app.app_context().push()
        
        # Sample data templates
        self.route_templates = [
            {
                'route_number': '1',
                'name': 'Downtown Express',
                'description': 'Fast service between downtown and university',
                'start_stop': 'Central Station',
                'end_stop': 'University Campus',
                'total_distance': 12.5,
                'estimated_duration': 35
            },
            {
                'route_number': '2',
                'name': 'Airport Shuttle',
                'description': 'Direct service to airport from city center',
                'start_stop': 'City Center',
                'end_stop': 'Airport Terminal',
                'total_distance': 18.2,
                'estimated_duration': 45
            },
            {
                'route_number': '3',
                'name': 'Suburban Loop',
                'description': 'Circular route through suburban areas',
                'start_stop': 'Mall Plaza',
                'end_stop': 'Mall Plaza',
                'total_distance': 25.8,
                'estimated_duration': 65
            },
            {
                'route_number': '4',
                'name': 'Hospital Line',
                'description': 'Service connecting major hospitals',
                'start_stop': 'General Hospital',
                'end_stop': 'Children\'s Hospital',
                'total_distance': 8.7,
                'estimated_duration': 25
            },
            {
                'route_number': '5',
                'name': 'Industrial Zone',
                'description': 'Service to industrial and business districts',
                'start_stop': 'Business District',
                'end_stop': 'Industrial Park',
                'total_distance': 15.3,
                'estimated_duration': 40
            }
        ]
        
        self.bus_models = [
            {'model': 'Mercedes Citaro', 'capacity': 80, 'year': 2020},
            {'model': 'Volvo 7900', 'capacity': 85, 'year': 2019},
            {'model': 'Scania Citywide', 'capacity': 90, 'year': 2021},
            {'model': 'MAN Lion\'s City', 'capacity': 75, 'year': 2018},
            {'model': 'Iveco Urbanway', 'capacity': 70, 'year': 2022}
        ]
        
        self.stop_names = [
            'Central Station', 'University Campus', 'City Center', 'Airport Terminal',
            'Mall Plaza', 'General Hospital', 'Children\'s Hospital', 'Business District',
            'Industrial Park', 'Shopping Center', 'Sports Complex', 'Library',
            'Post Office', 'Bank District', 'Residential Area', 'Park & Ride',
            'Train Station', 'Bus Terminal', 'Market Square', 'Government Building'
        ]
    
    def generate_all_data(self, days_back=30):
        """Generate all synthetic data"""
        print("Starting synthetic data generation...")
        
        # Clear existing data
        self.clear_existing_data()
        
        # Generate base data
        routes = self.generate_routes()
        buses = self.generate_buses(routes)
        stops = self.generate_stops(routes)
        
        # Generate historical data
        trips = self.generate_trips(routes, buses, days_back)
        events = self.generate_events(trips, stops, days_back)
        
        # Generate predictions
        predictions = self.generate_predictions(trips, stops)
        
        # Save to CSV files
        self.save_to_csv()
        
        print(f"Generated synthetic data:")
        print(f"- {len(routes)} routes")
        print(f"- {len(buses)} buses")
        print(f"- {len(stops)} stops")
        print(f"- {len(trips)} trips")
        print(f"- {len(events)} events")
        print(f"- {len(predictions)} predictions")
        print("Synthetic data generation completed!")
    
    def clear_existing_data(self):
        """Clear existing data from database"""
        print("Clearing existing data...")
        
        # Delete in reverse order of dependencies
        db.session.query(Prediction).delete()
        db.session.query(Event).delete()
        db.session.query(Trip).delete()
        db.session.query(RouteStop).delete()
        db.session.query(Bus).delete()
        db.session.query(Route).delete()
        
        db.session.commit()
        print("Existing data cleared.")
    
    def generate_routes(self):
        """Generate sample routes"""
        print("Generating routes...")
        
        routes = []
        for template in self.route_templates:
            route = Route(
                route_number=template['route_number'],
                name=template['name'],
                description=template['description'],
                start_stop=template['start_stop'],
                end_stop=template['end_stop'],
                total_distance=template['total_distance'],
                estimated_duration=template['estimated_duration'],
                is_active=True
            )
            db.session.add(route)
            routes.append(route)
        
        db.session.commit()
        return routes
    
    def generate_buses(self, routes):
        """Generate sample buses"""
        print("Generating buses...")
        
        buses = []
        bus_counter = 1
        
        for route in routes:
            # Generate 3-5 buses per route
            num_buses = random.randint(3, 5)
            
            for i in range(num_buses):
                bus_model = random.choice(self.bus_models)
                bus = Bus(
                    bus_number=f"{route.route_number}-{bus_counter:03d}",
                    license_plate=f"BUS{random.randint(1000, 9999)}",
                    capacity=bus_model['capacity'],
                    model=bus_model['model'],
                    year=bus_model['year'],
                    route_id=route.id,
                    is_active=True
                )
                db.session.add(bus)
                buses.append(bus)
                bus_counter += 1
        
        db.session.commit()
        return buses
    
    def generate_stops(self, routes):
        """Generate stops for each route"""
        print("Generating stops...")
        
        all_stops = []
        
        for route in routes:
            # Generate 8-15 stops per route
            num_stops = random.randint(8, 15)
            
            # Create stops along the route
            for i in range(num_stops):
                # Generate coordinates along a path
                lat = 40.7128 + (i * 0.01) + random.uniform(-0.005, 0.005)
                lng = -74.0060 + (i * 0.01) + random.uniform(-0.005, 0.005)
                
                stop = RouteStop(
                    route_id=route.id,
                    stop_name=random.choice(self.stop_names),
                    stop_code=f"{route.route_number}{i+1:02d}",
                    latitude=lat,
                    longitude=lng,
                    sequence_order=i + 1,
                    estimated_time_from_start=i * random.randint(3, 8)
                )
                db.session.add(stop)
                all_stops.append(stop)
        
        db.session.commit()
        return all_stops
    
    def generate_trips(self, routes, buses, days_back):
        """Generate historical trips"""
        print("Generating trips...")
        
        trips = []
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        for route in routes:
            route_buses = [b for b in buses if b.route_id == route.id]
            
            # Generate trips for each day
            for day in range(days_back):
                current_date = start_date + timedelta(days=day)
                
                # Skip weekends for some routes
                if current_date.weekday() >= 5 and random.random() < 0.3:
                    continue
                
                # Generate trips throughout the day
                for hour in range(6, 22):  # 6 AM to 10 PM
                    # More frequent during peak hours
                    if 7 <= hour <= 9 or 17 <= hour <= 19:
                        frequency = 15  # Every 15 minutes
                    else:
                        frequency = 30  # Every 30 minutes
                    
                    for minute in range(0, 60, frequency):
                        trip_time = current_date.replace(hour=hour, minute=minute, second=0)
                        
                        # Skip if trip time is in the future
                        if trip_time > datetime.utcnow():
                            continue
                        
                        bus = random.choice(route_buses)
                        
                        # Calculate actual start time with some delay
                        delay_minutes = random.randint(0, 10) if random.random() < 0.2 else 0
                        actual_start_time = trip_time + timedelta(minutes=delay_minutes)
                        
                        # Calculate end time
                        end_time = trip_time + timedelta(minutes=route.estimated_duration)
                        actual_end_time = actual_start_time + timedelta(minutes=route.estimated_duration)
                        
                        # Determine status
                        if actual_end_time < datetime.utcnow():
                            status = 'completed'
                        elif actual_start_time < datetime.utcnow():
                            status = 'in_progress'
                        else:
                            status = 'scheduled'
                        
                        trip = Trip(
                            route_id=route.id,
                            bus_id=bus.id,
                            scheduled_start_time=trip_time,
                            actual_start_time=actual_start_time if status != 'scheduled' else None,
                            scheduled_end_time=end_time,
                            actual_end_time=actual_end_time if status == 'completed' else None,
                            status=status
                        )
                        db.session.add(trip)
                        trips.append(trip)
        
        db.session.commit()
        return trips
    
    def generate_events(self, trips, stops, days_back):
        """Generate GPS and occupancy events"""
        print("Generating events...")
        
        events = []
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        for trip in trips:
            if trip.status in ['completed', 'in_progress']:
                # Get route stops
                route_stops = [s for s in stops if s.route_id == trip.route_id]
                route_stops.sort(key=lambda x: x.sequence_order)
                
                if not route_stops:
                    continue
                
                # Generate events along the route
                start_time = trip.actual_start_time or trip.scheduled_start_time
                duration_minutes = trip.route.estimated_duration
                
                # Generate GPS events every 2-5 minutes
                event_interval = random.randint(2, 5)
                
                for minute in range(0, duration_minutes, event_interval):
                    event_time = start_time + timedelta(minutes=minute)
                    
                    # Skip future events
                    if event_time > datetime.utcnow():
                        break
                    
                    # Calculate position along route
                    progress = minute / duration_minutes
                    stop_index = int(progress * (len(route_stops) - 1))
                    
                    if stop_index < len(route_stops):
                        current_stop = route_stops[stop_index]
                        next_stop = route_stops[min(stop_index + 1, len(route_stops) - 1)]
                        
                        # Interpolate position between stops
                        lat = current_stop.latitude + (next_stop.latitude - current_stop.latitude) * (progress % 1)
                        lng = current_stop.longitude + (next_stop.longitude - current_stop.longitude) * (progress % 1)
                        
                        # Add some noise to position
                        lat += random.uniform(-0.001, 0.001)
                        lng += random.uniform(-0.001, 0.001)
                        
                        # Generate speed
                        speed = random.uniform(15, 35)  # km/h
                        
                        # Generate heading
                        heading = random.uniform(0, 360)
                        
                        # GPS event
                        gps_event = Event(
                            trip_id=trip.id,
                            bus_id=trip.bus_id,
                            event_type='gps_update',
                            timestamp=event_time,
                            latitude=lat,
                            longitude=lng,
                            speed=speed,
                            heading=heading
                        )
                        db.session.add(gps_event)
                        events.append(gps_event)
                        
                        # Generate occupancy event (less frequent)
                        if random.random() < 0.3:
                            bus = next(b for b in trip.route.buses if b.id == trip.bus_id)
                            occupancy_count = random.randint(0, bus.capacity)
                            occupancy_percentage = (occupancy_count / bus.capacity) * 100
                            
                            occupancy_event = Event(
                                trip_id=trip.id,
                                bus_id=trip.bus_id,
                                event_type='occupancy_update',
                                timestamp=event_time,
                                occupancy_count=occupancy_count,
                                occupancy_percentage=occupancy_percentage
                            )
                            db.session.add(occupancy_event)
                            events.append(occupancy_event)
                        
                        # Generate delay event (occasionally)
                        if random.random() < 0.05:  # 5% chance
                            delay_minutes = random.randint(1, 15)
                            delay_reasons = ['traffic', 'passenger_loading', 'mechanical', 'weather']
                            
                            delay_event = Event(
                                trip_id=trip.id,
                                bus_id=trip.bus_id,
                                event_type='delay',
                                timestamp=event_time,
                                delay_minutes=delay_minutes,
                                data={'reason': random.choice(delay_reasons)}
                            )
                            db.session.add(delay_event)
                            events.append(delay_event)
        
        db.session.commit()
        return events
    
    def generate_predictions(self, trips, stops):
        """Generate sample predictions"""
        print("Generating predictions...")
        
        predictions = []
        
        # Generate predictions for recent trips
        recent_trips = [t for t in trips if t.scheduled_start_time >= datetime.utcnow() - timedelta(days=1)]
        
        for trip in recent_trips[:50]:  # Limit to 50 trips
            route_stops = [s for s in stops if s.route_id == trip.route_id]
            
            # Generate predictions for some stops
            for stop in route_stops[::2]:  # Every other stop
                # Calculate predicted arrival time
                base_time = trip.scheduled_start_time
                time_to_stop = stop.estimated_time_from_start or 0
                predicted_time = base_time + timedelta(minutes=time_to_stop)
                
                # Add some randomness
                predicted_time += timedelta(minutes=random.randint(-5, 10))
                
                prediction = Prediction(
                    trip_id=trip.id,
                    stop_id=stop.id,
                    predicted_arrival_time=predicted_time,
                    confidence_score=random.uniform(0.7, 0.95),
                    model_version='1.0',
                    features_used={'route_id': trip.route_id, 'stop_sequence': stop.sequence_order}
                )
                db.session.add(prediction)
                predictions.append(prediction)
        
        db.session.commit()
        return predictions
    
    def save_to_csv(self):
        """Save generated data to CSV files"""
        print("Saving data to CSV files...")
        
        # Ensure data directory exists
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/clean', exist_ok=True)
        
        # Save routes
        routes_df = pd.read_sql('SELECT * FROM routes', db.engine)
        routes_df.to_csv('data/raw/routes.csv', index=False)
        
        # Save buses
        buses_df = pd.read_sql('SELECT * FROM buses', db.engine)
        buses_df.to_csv('data/raw/buses.csv', index=False)
        
        # Save route stops
        stops_df = pd.read_sql('SELECT * FROM route_stops', db.engine)
        stops_df.to_csv('data/raw/route_stops.csv', index=False)
        
        # Save trips
        trips_df = pd.read_sql('SELECT * FROM trips', db.engine)
        trips_df.to_csv('data/raw/trips.csv', index=False)
        
        # Save events
        events_df = pd.read_sql('SELECT * FROM events', db.engine)
        events_df.to_csv('data/raw/events.csv', index=False)
        
        # Save predictions
        predictions_df = pd.read_sql('SELECT * FROM predictions', db.engine)
        predictions_df.to_csv('data/raw/predictions.csv', index=False)
        
        print("Data saved to CSV files in data/raw/ directory")

def main():
    """Main function to run the synthetic data generator"""
    generator = SyntheticDataGenerator()
    
    # Generate data for the last 30 days
    generator.generate_all_data(days_back=30)
    
    print("\nSynthetic data generation completed successfully!")
    print("You can now run the application with realistic test data.")

if __name__ == '__main__':
    main()
