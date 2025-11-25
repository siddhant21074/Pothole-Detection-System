import folium
import os
from datetime import datetime

class MapProcessor:
    def __init__(self, map_file='data/maps/offline_map.osm.pbf'):
        self.map_file = map_file
        self.map = None
        self.pothole_locations = []
        # Don't initialize map here, wait for first location update

    def initialize_map(self, center_lat=None, center_lon=None, zoom=15):
        """Initialize the map with optional center coordinates"""
        # Default to center of India if no coordinates provided
        if center_lat is None or center_lon is None:
            center_lat, center_lon = 20.5937, 78.9629  # Center of India
        
        self.map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='OpenStreetMap',
            control_scale=True
        )
        return self.map

    def update_map_center(self, lat, lon):
        """Update the map center if map is already initialized"""
        if self.map is not None:
            self.map.location = [lat, lon]

    def add_pothole_marker(self, lat, lon, confidence, timestamp):
        """Add a pothole marker to the map"""
        if self.map is None:
            self.initialize_map(lat, lon)
            
        marker = {
            'lat': lat,
            'lon': lon,
            'confidence': confidence,
            'timestamp': timestamp
        }
        self.pothole_locations.append(marker)
        
        # Add marker to map
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=f'Pothole (Confidence: {confidence:.2f})'
        ).add_to(self.map)
        
        return marker

    def save_map(self, output_file='templates/map.html'):
        """Save the current map to an HTML file"""
        if self.map:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            self.map.save(output_file)
            return True
        return False

    def save_potholes_to_geojson(self, output_file='data/potholes.geojson'):
        """Save pothole locations to a GeoJSON file"""
        features = []
        for pothole in self.pothole_locations:
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [pothole['lon'], pothole['lat']]
                },
                'properties': {
                    'confidence': pothole['confidence'],
                    'timestamp': pothole['timestamp']
                }
            })
            
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)