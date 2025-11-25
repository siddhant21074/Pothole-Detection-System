import cv2
import threading
import time
import logging
from typing import Optional, Tuple
import numpy as np

class CameraThread:
    def __init__(self, src=0, width=1280, height=720, fps=30, api_preference=None):
        """Initialize camera with basic settings."""
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.api_preference = api_preference
        
        # Camera properties
        self.stream = None
        self.frame = None
        self.grabbed = False
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        
        # Initialize logger
        self.logger = logging.getLogger('CameraThread')
        self.logger.setLevel(logging.INFO)
        
        # Start camera thread
        self.start()

    def _init_camera(self) -> bool:
        """Initialize camera with basic error handling."""
        try:
            # Release previous stream if exists
            if self.stream is not None:
                self.stream.release()
            
            # Try to open camera
            if self.api_preference is not None:
                self.stream = cv2.VideoCapture(self.src, self.api_preference)
            else:
                self.stream = cv2.VideoCapture(self.src)
            
            if not self.stream.isOpened():
                return False

            # Set camera properties
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.stream.set(cv2.CAP_PROP_FPS, self.fps)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            
            # Test read
            for _ in range(3):
                ret, frame = self.stream.read()
                if ret and frame is not None:
                    self.frame = frame
                    self.grabbed = True
                    return True
                time.sleep(0.1)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Camera init error: {e}")
            return False

    def start(self):
        """Start the camera thread."""
        if self.running:
            return
            
        if not self._init_camera():
            self.logger.error("Failed to initialize camera")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Main camera update loop."""
        while self.running:
            try:
                ret, frame = self.stream.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                    
                with self.lock:
                    self.frame = frame
                    self.grabbed = True
                    
                # Small delay to maintain target FPS
                time.sleep(max(0, 1.0/self.fps - 0.01))
                
            except Exception as e:
                self.logger.error(f"Update error: {e}")
                time.sleep(0.5)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the current frame."""
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        """Release camera resources."""
        self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        if self.stream is not None:
            try:
                self.stream.release()
            except:
                pass
            self.stream = None
            
        self.frame = None
        self.grabbed = False

    def __del__(self):
        """Ensure resources are released."""
        self.release()