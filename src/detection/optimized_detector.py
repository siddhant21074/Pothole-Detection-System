# In src/detection/optimized_detector.py
import time
import cv2
import numpy as np
import torch
import logging
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)

class OptimizedPotholeDetector:
    def __init__(self, model_path: str = 'models/optimized/yolov8s_pothole_optimized/weights/best.pt',
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45, frame_skip: int = 1):
        """
        Initialize the optimized pothole detector.
        
        Args:
            model_path: Path to the YOLO model weights
            conf_threshold: Confidence threshold for detections (lowered for better recall)
            iou_threshold: IoU threshold for NMS
            frame_skip: Number of frames to skip between detections (1 = every frame)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)  # Use deque for efficient operations
        self.input_size = 640
        self.last_detections = []
        self.last_frame = None
        
        # Performance optimization flags
        self.use_half = False  # FP16 inference
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            
            # Optimize model for inference
            if self.device == 'cuda':
                self.model.to('cuda')
                # Try to use half precision (FP16) on GPU
                try:
                    self.model.model.half()
                    self.use_half = True
                    logger.info("Using FP16 (half precision) for faster inference")
                except:
                    logger.info("FP16 not available, using FP32")
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Confidence threshold: {self.conf_threshold}")
            logger.info(f"IOU threshold: {self.iou_threshold}")
            
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}")
            logger.info("Attempting to use default YOLOv8n model...")
            try:
                self.model = YOLO('yolov8n.pt')
                logger.warning("Using default YOLOv8n model - detections may be less accurate")
            except Exception as e:
                raise RuntimeError(f"Failed to load any model: {e}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better detection"""
        # Enhance contrast for better detection
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Detect potholes in the frame using YOLO with optimizations."""
        if frame is None or frame.size == 0:
            return [], frame
            
        self.frame_count += 1
        
        # Skip frames for performance if configured
        if self.frame_skip > 1 and self.frame_count % self.frame_skip != 0:
            if self.last_frame is not None:
                return self.last_detections, self.last_frame
            return [], frame
        
        try:
            start = time.time()
            
            # Preprocess frame for better detection
            processed = self.preprocess_frame(frame)
            
            # Run inference with YOLO
            with torch.no_grad():  # Disable gradient computation for inference
                results = self.model(
                    processed,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    half=self.use_half,
                    device=self.device
                )
            
            # Extract detections from results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = box
                        
                        # Filter out very small or very large boxes (likely false positives)
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        frame_area = frame.shape[0] * frame.shape[1]
                        
                        # Skip if box is less than 0.1% or more than 80% of frame
                        if area < frame_area * 0.001 or area > frame_area * 0.8:
                            continue
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class_id': 0,
                            'class_name': 'pothole',
                            'timestamp': time.time()
                        })
            
            # Draw detections on frame with enhanced visualization
            display_frame = frame.copy()
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                conf = det['confidence']
                
                # Color based on confidence: red (high) -> yellow (medium) -> orange (low)
                if conf >= 0.7:
                    color = (0, 0, 255)  # Red for high confidence
                    thickness = 3
                elif conf >= 0.5:
                    color = (0, 165, 255)  # Orange for medium
                    thickness = 2
                else:
                    color = (0, 255, 255)  # Yellow for low
                    thickness = 2
                
                # Draw bounding box with rounded corners effect
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw corner markers for better visibility
                corner_length = 15
                cv2.line(display_frame, (x1, y1), (x1 + corner_length, y1), color, thickness + 1)
                cv2.line(display_frame, (x1, y1), (x1, y1 + corner_length), color, thickness + 1)
                cv2.line(display_frame, (x2, y1), (x2 - corner_length, y1), color, thickness + 1)
                cv2.line(display_frame, (x2, y1), (x2, y1 + corner_length), color, thickness + 1)
                cv2.line(display_frame, (x1, y2), (x1 + corner_length, y2), color, thickness + 1)
                cv2.line(display_frame, (x1, y2), (x1, y2 - corner_length), color, thickness + 1)
                cv2.line(display_frame, (x2, y2), (x2 - corner_length, y2), color, thickness + 1)
                cv2.line(display_frame, (x2, y2), (x2, y2 - corner_length), color, thickness + 1)
                
                # Draw label with background
                label = f"Pothole {conf:.2%}"
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw background rectangle for text
                cv2.rectangle(display_frame, 
                            (x1, y1 - text_h - baseline - 8), 
                            (x1 + text_w + 8, y1), 
                            color, -1)
                
                # Draw text
                cv2.putText(display_frame, label, (x1 + 4, y1 - baseline - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update FPS calculation
            frame_time = time.time() - start
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) > 0:
                avg_time = sum(self.frame_times) / len(self.frame_times)
                self.fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Draw info panel
            panel_height = 100
            panel = np.zeros((panel_height, display_frame.shape[1], 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)  # Dark gray background
            
            # Add info text
            info_texts = [
                f"FPS: {self.fps:.1f}",
                f"Detections: {len(detections)}",
                f"Frame: {self.frame_count}",
                f"Device: {self.device.upper()}"
            ]
            
            y_offset = 25
            for i, text in enumerate(info_texts):
                cv2.putText(panel, text, (10, y_offset + i * 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Combine panel with frame
            display_frame = np.vstack([panel, display_frame])
            
            # Cache results
            self.last_detections = detections
            self.last_frame = display_frame
            
            return detections, display_frame
            
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            # Return frame with error message
            error_frame = frame.copy()
            cv2.putText(error_frame, f"Detection Error: {str(e)[:50]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return [], error_frame
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'device': self.device,
            'use_half': self.use_half,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold
        }