import math
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth in kilometers.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in decimal degrees
        lat2, lon2: Latitude and longitude of second point in decimal degrees
    
    Returns:
        Distance in kilometers
    """
    try:
        # Convert decimal degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        earth_radius_km = 6371
        
        return earth_radius_km * c
        
    except Exception as e:
        logger.error(f"Error calculating distance: {str(e)}")
        return 0.0

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing (direction) from point 1 to point 2.
    
    Args:
        lat1, lon1: Starting point coordinates
        lat2, lon2: Ending point coordinates
    
    Returns:
        Bearing in degrees (0-360)
    """
    try:
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360 degrees
        return (bearing_deg + 360) % 360
        
    except Exception as e:
        logger.error(f"Error calculating bearing: {str(e)}")
        return 0.0

def interpolate_position(lat1: float, lon1: float, lat2: float, lon2: float, 
                       fraction: float) -> Tuple[float, float]:
    """
    Interpolate a position between two points.
    
    Args:
        lat1, lon1: Starting point coordinates
        lat2, lon2: Ending point coordinates
        fraction: Fraction of the way from point 1 to point 2 (0.0 to 1.0)
    
    Returns:
        Tuple of (latitude, longitude) for the interpolated position
    """
    try:
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Calculate the angular distance
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Handle longitude wrapping
        if abs(dlon) > math.pi:
            if dlon > 0:
                dlon -= 2 * math.pi
            else:
                dlon += 2 * math.pi
        
        # Interpolate
        lat_interp_rad = lat1_rad + fraction * dlat
        lon_interp_rad = lon1_rad + fraction * dlon
        
        # Convert back to degrees
        return math.degrees(lat_interp_rad), math.degrees(lon_interp_rad)
        
    except Exception as e:
        logger.error(f"Error interpolating position: {str(e)}")
        return lat1, lon1

def format_time(dt: datetime, format_type: str = 'display') -> str:
    """
    Format datetime for display.
    
    Args:
        dt: Datetime object
        format_type: Type of formatting ('display', 'api', 'short')
    
    Returns:
        Formatted time string
    """
    try:
        if format_type == 'display':
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        elif format_type == 'api':
            return dt.isoformat()
        elif format_type == 'short':
            return dt.strftime('%H:%M')
        elif format_type == 'date':
            return dt.strftime('%Y-%m-%d')
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S')
            
    except Exception as e:
        logger.error(f"Error formatting time: {str(e)}")
        return str(dt)

def calculate_travel_time(distance_km: float, avg_speed_kmh: float) -> int:
    """
    Calculate travel time in minutes.
    
    Args:
        distance_km: Distance in kilometers
        avg_speed_kmh: Average speed in km/h
    
    Returns:
        Travel time in minutes
    """
    try:
        if avg_speed_kmh <= 0:
            return 0
        
        time_hours = distance_km / avg_speed_kmh
        time_minutes = int(time_hours * 60)
        
        return max(1, time_minutes)  # Minimum 1 minute
        
    except Exception as e:
        logger.error(f"Error calculating travel time: {str(e)}")
        return 0

def calculate_eta(current_time: datetime, distance_km: float, 
                 avg_speed_kmh: float) -> datetime:
    """
    Calculate estimated time of arrival.
    
    Args:
        current_time: Current time
        distance_km: Remaining distance in kilometers
        avg_speed_kmh: Average speed in km/h
    
    Returns:
        Estimated arrival time
    """
    try:
        travel_minutes = calculate_travel_time(distance_km, avg_speed_kmh)
        return current_time + timedelta(minutes=travel_minutes)
        
    except Exception as e:
        logger.error(f"Error calculating ETA: {str(e)}")
        return current_time

def is_peak_hour(dt: datetime) -> bool:
    """
    Check if the given time is during peak hours.
    
    Args:
        dt: Datetime object
    
    Returns:
        True if during peak hours (7-9 AM or 5-7 PM)
    """
    try:
        hour = dt.hour
        return (7 <= hour <= 9) or (17 <= hour <= 19)
        
    except Exception as e:
        logger.error(f"Error checking peak hour: {str(e)}")
        return False

def is_weekend(dt: datetime) -> bool:
    """
    Check if the given date is a weekend.
    
    Args:
        dt: Datetime object
    
    Returns:
        True if Saturday or Sunday
    """
    try:
        return dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday
        
    except Exception as e:
        logger.error(f"Error checking weekend: {str(e)}")
        return False

def calculate_headway(trip1_time: datetime, trip2_time: datetime) -> int:
    """
    Calculate headway between two trips in minutes.
    
    Args:
        trip1_time: Time of first trip
        trip2_time: Time of second trip
    
    Returns:
        Headway in minutes
    """
    try:
        time_diff = trip2_time - trip1_time
        return int(time_diff.total_seconds() / 60)
        
    except Exception as e:
        logger.error(f"Error calculating headway: {str(e)}")
        return 0

def calculate_delay(scheduled_time: datetime, actual_time: datetime) -> int:
    """
    Calculate delay in minutes.
    
    Args:
        scheduled_time: Scheduled time
        actual_time: Actual time
    
    Returns:
        Delay in minutes (positive for late, negative for early)
    """
    try:
        time_diff = actual_time - scheduled_time
        return int(time_diff.total_seconds() / 60)
        
    except Exception as e:
        logger.error(f"Error calculating delay: {str(e)}")
        return 0

def calculate_occupancy_percentage(count: int, capacity: int) -> float:
    """
    Calculate occupancy percentage.
    
    Args:
        count: Current passenger count
        capacity: Bus capacity
    
    Returns:
        Occupancy percentage (0-100)
    """
    try:
        if capacity <= 0:
            return 0.0
        
        percentage = (count / capacity) * 100
        return min(100.0, max(0.0, percentage))
        
    except Exception as e:
        logger.error(f"Error calculating occupancy percentage: {str(e)}")
        return 0.0

def calculate_route_efficiency(actual_time: datetime, scheduled_time: datetime, 
                             distance_km: float) -> float:
    """
    Calculate route efficiency score.
    
    Args:
        actual_time: Actual completion time
        scheduled_time: Scheduled completion time
        distance_km: Route distance in kilometers
    
    Returns:
        Efficiency score (0-1, higher is better)
    """
    try:
        delay_minutes = abs(calculate_delay(scheduled_time, actual_time))
        
        # Base efficiency on delay and distance
        if distance_km <= 0:
            return 0.0
        
        # Normalize delay by distance (minutes per km)
        delay_per_km = delay_minutes / distance_km
        
        # Efficiency decreases with delay per km
        efficiency = max(0.0, 1.0 - (delay_per_km / 2.0))  # 2 minutes per km is 0 efficiency
        
        return min(1.0, efficiency)
        
    except Exception as e:
        logger.error(f"Error calculating route efficiency: {str(e)}")
        return 0.0

def calculate_bunching_score(trips: List[Dict]) -> float:
    """
    Calculate bunching score for a set of trips.
    
    Args:
        trips: List of trip dictionaries with 'time' and 'position' keys
    
    Returns:
        Bunching score (0-1, higher means more bunching)
    """
    try:
        if len(trips) < 2:
            return 0.0
        
        # Sort trips by time
        sorted_trips = sorted(trips, key=lambda x: x['time'])
        
        bunching_score = 0.0
        total_pairs = 0
        
        for i in range(len(sorted_trips) - 1):
            trip1 = sorted_trips[i]
            trip2 = sorted_trips[i + 1]
            
            # Calculate time difference
            time_diff = (trip2['time'] - trip1['time']).total_seconds() / 60  # minutes
            
            # Calculate position difference
            if 'position' in trip1 and 'position' in trip2:
                pos1 = trip1['position']
                pos2 = trip2['position']
                distance = calculate_distance(pos1[0], pos1[1], pos2[0], pos2[1])
                
                # Bunching occurs when buses are close in time and space
                if time_diff <= 5 and distance <= 1.0:  # Within 5 minutes and 1 km
                    bunching_score += 1.0
            
            total_pairs += 1
        
        return bunching_score / total_pairs if total_pairs > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating bunching score: {str(e)}")
        return 0.0

def calculate_service_reliability(trips: List[Dict]) -> float:
    """
    Calculate service reliability score.
    
    Args:
        trips: List of trip dictionaries with 'scheduled_time' and 'actual_time' keys
    
    Returns:
        Reliability score (0-1, higher is better)
    """
    try:
        if not trips:
            return 0.0
        
        on_time_count = 0
        total_trips = len(trips)
        
        for trip in trips:
            if 'scheduled_time' in trip and 'actual_time' in trip:
                delay = abs(calculate_delay(trip['scheduled_time'], trip['actual_time']))
                if delay <= 5:  # Within 5 minutes is considered on-time
                    on_time_count += 1
        
        return on_time_count / total_trips if total_trips > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating service reliability: {str(e)}")
        return 0.0

def calculate_average_speed(events: List[Dict]) -> float:
    """
    Calculate average speed from GPS events.
    
    Args:
        events: List of event dictionaries with 'speed' key
    
    Returns:
        Average speed in km/h
    """
    try:
        if not events:
            return 0.0
        
        speeds = [event.get('speed', 0) for event in events if event.get('speed', 0) > 0]
        
        if not speeds:
            return 0.0
        
        return sum(speeds) / len(speeds)
        
    except Exception as e:
        logger.error(f"Error calculating average speed: {str(e)}")
        return 0.0

def calculate_average_occupancy(events: List[Dict]) -> float:
    """
    Calculate average occupancy from occupancy events.
    
    Args:
        events: List of event dictionaries with 'occupancy_percentage' key
    
    Returns:
        Average occupancy percentage
    """
    try:
        if not events:
            return 0.0
        
        occupancies = [event.get('occupancy_percentage', 0) for event in events 
                      if event.get('occupancy_percentage') is not None]
        
        if not occupancies:
            return 0.0
        
        return sum(occupancies) / len(occupancies)
        
    except Exception as e:
        logger.error(f"Error calculating average occupancy: {str(e)}")
        return 0.0

def generate_time_series_data(start_time: datetime, end_time: datetime, 
                            interval_minutes: int = 15) -> List[datetime]:
    """
    Generate a list of datetime objects for time series analysis.
    
    Args:
        start_time: Start time
        end_time: End time
        interval_minutes: Interval between points in minutes
    
    Returns:
        List of datetime objects
    """
    try:
        time_points = []
        current_time = start_time
        
        while current_time <= end_time:
            time_points.append(current_time)
            current_time += timedelta(minutes=interval_minutes)
        
        return time_points
        
    except Exception as e:
        logger.error(f"Error generating time series data: {str(e)}")
        return []

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with statistics
    """
    try:
        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        values_array = np.array(values)
        
        return {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'median': float(np.median(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array))
        }
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }

def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Validate if coordinates are within valid ranges.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
    
    Returns:
        True if coordinates are valid
    """
    try:
        return (-90 <= latitude <= 90) and (-180 <= longitude <= 180)
        
    except Exception as e:
        logger.error(f"Error validating coordinates: {str(e)}")
        return False

def validate_time_range(start_time: datetime, end_time: datetime) -> bool:
    """
    Validate if time range is reasonable.
    
    Args:
        start_time: Start time
        end_time: End time
    
    Returns:
        True if time range is valid
    """
    try:
        if start_time >= end_time:
            return False
        
        # Check if range is not too long (more than 24 hours)
        time_diff = end_time - start_time
        if time_diff > timedelta(hours=24):
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating time range: {str(e)}")
        return False

def format_duration(minutes: int) -> str:
    """
    Format duration in minutes to human-readable string.
    
    Args:
        minutes: Duration in minutes
    
    Returns:
        Formatted duration string
    """
    try:
        if minutes < 60:
            return f"{minutes} min"
        else:
            hours = minutes // 60
            remaining_minutes = minutes % 60
            if remaining_minutes == 0:
                return f"{hours} hr"
            else:
                return f"{hours} hr {remaining_minutes} min"
                
    except Exception as e:
        logger.error(f"Error formatting duration: {str(e)}")
        return f"{minutes} min"

def format_distance(km: float) -> str:
    """
    Format distance in kilometers to human-readable string.
    
    Args:
        km: Distance in kilometers
    
    Returns:
        Formatted distance string
    """
    try:
        if km < 1:
            return f"{km * 1000:.0f} m"
        else:
            return f"{km:.1f} km"
            
    except Exception as e:
        logger.error(f"Error formatting distance: {str(e)}")
        return f"{km} km"

def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate percentile value from a list of numbers.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
    
    Returns:
        Percentile value
    """
    try:
        if not values:
            return 0.0
        
        values_array = np.array(values)
        return float(np.percentile(values_array, percentile))
        
    except Exception as e:
        logger.error(f"Error calculating percentile: {str(e)}")
        return 0.0

def detect_outliers(values: List[float], threshold: float = 2.0) -> List[int]:
    """
    Detect outliers in a list of values using z-score method.
    
    Args:
        values: List of numeric values
        threshold: Z-score threshold for outlier detection
    
    Returns:
        List of indices of outlier values
    """
    try:
        if len(values) < 3:
            return []
        
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            return []
        
        z_scores = np.abs((values_array - mean) / std)
        outlier_indices = np.where(z_scores > threshold)[0]
        
        return outlier_indices.tolist()
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        return []
