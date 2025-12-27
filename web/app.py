# In web/app.py
import os
import sys
import cv2
import time
import json
import torch
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin.db import Reference
from typing import Dict, Any, Optional, Tuple, List
from flask import Flask, render_template_string, Response, request, jsonify
from threading import Lock
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import your module
from src.detection.yolov8_pretrained_detector import YOLOv8PotholeDetector

app = Flask(__name__)

# Initialize Firebase
firebase_initialized = False

def init_firebase():
    global firebase_initialized
    try:
        # Path to your Firebase service account key
        cred_path = os.path.join(project_root, 'config', 'firebase-key.json')
        
        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://potholesdb-default-rtdb.firebaseio.com/'
        })
        firebase_initialized = True
        print("✓ Firebase initialized successfully")
    except Exception as e:
        print(f"✗ Firebase initialization failed: {e}")
        firebase_initialized = False

# Initialize Firebase
init_firebase()

# Thread-safe camera access
camera_lock = Lock()
frame_lock = Lock()

def save_pothole_to_firebase(lat: float, lon: float, confidence: float, timestamp: str):
    """Save pothole detection to Firebase Realtime Database"""
    print(f"\n=== Attempting to save to Firebase ===")
    print(f"Location: {lat}, {lon}")
    print(f"Confidence: {confidence}")
    print(f"Timestamp: {timestamp}")
    
    if not firebase_initialized:
        print("✗ Firebase not initialized, skipping save to database")
        return False
    
    try:
        # Get a reference to the 'coordinates' node
        ref = db.reference('coordinates')
        print("✓ Connected to Firebase 'coordinates' node")
        
        # Get the next available ID (find max ID and increment)
        all_coords = ref.get() or {}
        print(f"Found {len(all_coords)} existing coordinates")
        
        # Handle case where there are no coordinates yet
        if not all_coords:
            max_id = 100  # Start from 100 if no coordinates exist
        else:
            try:
                # Get all numeric keys and find max
                numeric_keys = [int(k) for k in all_coords.keys() if k.isdigit()]
                max_id = max(numeric_keys) + 1 if numeric_keys else 100
            except Exception as e:
                print(f"Error finding max ID: {e}")
                max_id = 100
        
        print(f"Using ID: {max_id}")
        
        # Prepare data with proper types
        pothole_data = {
            'id': str(max_id),
            'latitude': str(round(lat, 6)),
            'longitude': str(round(lon, 6)),
            'timestamp': timestamp,
            'confidence': str(round(float(confidence), 2))
        }
        
        print("Saving data:", pothole_data)
        
        # Save the data with the numeric ID as the key
        ref.child(str(max_id)).set(pothole_data)
        
        print(f"✓ Pothole data saved to Firebase (ID: {max_id})")
        return True
    except Exception as e:
        print(f"✗ Failed to save pothole to Firebase: {e}")
        import traceback
        traceback.print_exc()
        return False

# Global variables
detector = None
camera = None
current_location = {'lat': 20.5937, 'lon': 78.9629, 'accuracy': 50000}
potholes = {'type': 'FeatureCollection', 'features': []}
latest_frame = None

def init_detector():
    """Initialize the custom YOLOv8 detector"""
    global detector
    try:
        # Path to custom model
        model_path = os.path.join(project_root, 'models', 'optimized', 'yolov8s_pothole_optimized', 'weights', 'best.pt')
        
        if not os.path.exists(model_path):
            print(f"✗ Custom model not found at {model_path}")
            return False
            
        # Initialize YOLO model with custom weights
        detector = YOLO(model_path)
        detector.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set model parameters
        detector.overrides['conf'] = 0.4  # Confidence threshold
        detector.overrides['iou'] = 0.45  # NMS IoU threshold
        detector.overrides['agnostic_nms'] = True
        
        print("✓ Custom YOLOv8 Detector initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize custom YOLOv8 detector: {e}")
        detector = None
        return False

def init_camera():
    """Initialize the default camera with multiple fallback options"""
    global camera
    
    # Try different camera backends and indices
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto"),
    ]
    
    camera_indices = [0, 1, 2]
    
    for backend, backend_name in backends:
        for idx in camera_indices:
            try:
                print(f"Trying camera {idx} with {backend_name}...")
                cam = cv2.VideoCapture(idx, backend)
                
                if cam.isOpened():
                    # Configure camera
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cam.set(cv2.CAP_PROP_FPS, 30)
                    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test read
                    ret, frame = cam.read()
                    if ret and frame is not None:
                        camera = cam
                        print(f"✓ Camera {idx} initialized with {backend_name}")
                        print(f"  Resolution: {int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                        print(f"  FPS: {int(cam.get(cv2.CAP_PROP_FPS))}")
                        return True
                    else:
                        cam.release()
                        
            except Exception as e:
                print(f"✗ Failed to open camera {idx} with {backend_name}: {e}")
                continue
    
    print("✗ No camera found!")
    camera = None
    return False

def process_detection(frame, model, conf_threshold: float = 0.4) -> Tuple[List[Dict], np.ndarray]:
    """Process frame with YOLOv8 model and return detections and annotated frame"""
    # Make a copy of the frame for drawing
    annotated_frame = frame.copy()
    detections = []
    
    # Run inference
    results = model(frame, verbose=False)
    
    # Process results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes in (x1, y1, x2, y2) format
        confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            # Only process pothole class (class_id 0)
            if class_id != 0 or conf < conf_threshold:
                continue
                
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Add to detections
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class_id': int(class_id),
                'timestamp': datetime.now().isoformat()
            })
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"Pothole: {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + label_width, y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return detections, annotated_frame

def gen_frames():
    """Generate frames with pothole detection"""
    global latest_frame, potholes, current_location
    
    frame_count = 0
    last_detection_time = 0
    detection_interval = 0.1  # Run detection every 100ms
    
    while True:
        if camera is None:
            # Generate a placeholder frame
            placeholder = create_placeholder_frame()
            ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
            
        try:
            with camera_lock:
                success, frame = camera.read()
            
            if not success or frame is None:
                print("Failed to read frame, reinitializing camera...")
                init_camera()
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Run detection at intervals
            if detector and (current_time - last_detection_time) >= detection_interval:
                try:
                    detections, processed_frame = process_detection(frame, detector)
                    last_detection_time = current_time
                    
                    # Update potholes with location
                    if detections:
                        print(f"\n=== Detected {len(detections)} potholes ===")
                        for i, det in enumerate(detections, 1):
                            confidence = det['confidence']
                            timestamp = det['timestamp']
                            
                            print(f"\nPothole {i}:")
                            print(f"- Confidence: {confidence}")
                            print(f"- Timestamp: {timestamp}")
                            print(f"- Current Location: {current_location}")
                            
                            # Save to Firebase if confidence is high enough
                            if confidence >= 0.5:  # Only save high confidence detections
                                save_pothole_to_firebase(
                                    lat=current_location['lat'],
                                    lon=current_location['lon'],
                                    confidence=confidence,
                                    timestamp=timestamp
                                )
                                
                                # Add to potholes list for mapping
                                potholes['features'].append({
                                    'type': 'Feature',
                                    'geometry': {
                                        'type': 'Point',
                                        'coordinates': [current_location['lon'], current_location['lat']]
                                    },
                                    'properties': {
                                        'confidence': confidence,
                                        'timestamp': timestamp
                                    }
                                })
                    
                    # Update the latest frame with detections
                    latest_frame = processed_frame
                    
                except Exception as e:
                    print(f"Error during detection: {e}")
                    processed_frame = frame
            else:
                # Use the last processed frame if not time for new detection yet
                processed_frame = latest_frame if latest_frame is not None else frame
                
            # Add info overlay to the processed frame
            cv2.putText(processed_frame, f"Frame: {frame_count}", 
                       (10, processed_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Encode the processed frame
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Small delay for frame rate control
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Frame generation error: {e}")
            time.sleep(0.1)

def create_placeholder_frame():
    """Create a placeholder frame when camera is not available"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    text = "Camera Not Available"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (480 + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
    return frame

@app.route('/')
def index():
    # Get the directory where this script is located
    web_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(web_dir, 'templates', 'index.html')
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return f"index.html not found at: {html_path}", 404
    except Exception as e:
        return f"Error loading template: {str(e)}", 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_location', methods=['POST'])
def update_location():
    """Update current location from GPS"""
    global current_location
    try:
        data = request.get_json()
        current_location = {
            'lat': data.get('lat', current_location['lat']),
            'lon': data.get('lon', current_location['lon']),
            'accuracy': data.get('accuracy', current_location['accuracy'])
        }
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/get_potholes')
def get_potholes():
    """Get all detected potholes"""
    return jsonify(potholes)

@app.route('/camera_status')
def camera_status():
    """Check camera status"""
    status = {
        'camera_available': camera is not None and camera.isOpened(),
        'detector_available': detector is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    print("=" * 50)
    print("Pothole Detection System - Starting")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing camera...")
    camera_ok = init_camera()
    
    print("\n2. Initializing detector...")
    detector_ok = init_detector()
    
    print("\n" + "=" * 50)
    if camera_ok and detector_ok:
        print("✓ System ready!")
    elif camera_ok:
        print("⚠ System ready (no detector - only camera feed)")
    elif detector_ok:
        print("⚠ System ready (no camera - detection disabled)")
    else:
        print("⚠ System starting with limited functionality")
    print("=" * 50)
    print("\n🌐 Open http://127.0.0.1:5000 in your browser\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        print("\nShutting down...")
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()