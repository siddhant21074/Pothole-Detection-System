# In src/utils/gps_tracker.py
from datetime import datetime
import time
import json

class GPSTracker:
    def __init__(self):
        self.current_lat = None
        self.current_lon = None
        self.accuracy = None
        self.last_updated = None
        self.location_updated = False
        self.last_error = None
        self.location_attempts = 0
        
        # For debugging
        print("GPSTracker initialized")

    def update_location(self, lat, lon, accuracy=None):
        """Update the current location with validation"""
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            print(f"Invalid coordinates received: lat={lat}, lon={lon}")
            self.last_error = f"Invalid coordinates: lat={lat}, lon={lon}"
            return False
            
        self.current_lat = lat
        self.current_lon = lon
        self.accuracy = accuracy
        self.last_updated = datetime.now()
        self.location_updated = True
        print(f"Location updated: {lat}, {lon} (Accuracy: {accuracy}m)")
        return True

    def get_current_location(self):
        """Return the last known location or try to get from browser"""
        if self.current_lat is not None and self.current_lon is not None:
            return self.current_lat, self.current_lon
            
        # If no location yet, try to get from browser
        print("No cached location, attempting to get from browser...")
        return self._get_browser_location()

    def get_location_status(self):
        """Get the status of the location data"""
        return {
            'has_location': self.current_lat is not None,
            'lat': self.current_lat,
            'lon': self.current_lon,
            'accuracy': self.accuracy,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'error': self.last_error,
            'attempts': self.location_attempts
        }
        
    def _get_browser_location(self):
        """Attempt to get location from browser"""
        self.location_attempts += 1
        try:
            # This will trigger the browser's geolocation API
            # The actual location will be set via the /update_location endpoint
            return None, None
        except Exception as e:
            self.last_error = f"Browser location error: {str(e)}"
            print(self.last_error)
            return None, None