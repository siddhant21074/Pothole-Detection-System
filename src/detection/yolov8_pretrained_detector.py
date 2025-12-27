import os
import cv2
import numpy as np
import torch
import logging
from ultralytics import YOLO
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class YOLOv8PotholeDetector:
    def __init__(self, model_size: str = 'n', conf_threshold: float = 0.4, iou_threshold: float = 0.5):
        """
        Initialize the YOLOv8 pothole detector with the trained model.

        Args:
            model_size: Size of YOLOv8 model (n, s, m, l, x). Only used if optimizer.pt is not found.
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        
        # Initialize the model
        self._initialize_model()

    def _initialize_model(self):
        """Load the trained model from optimizer.pt"""
        try:
            # Path to the optimizer.pt in the root directory
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'optimizer.pt')
            
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                logger.info(f"✓ Loaded trained model from {model_path}")
                logger.info(f"Using device: {self.device}")
            else:
                raise FileNotFoundError(f"Trained model not found at {model_path}")
            
            # Set model to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the input frame for better detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed frame
        """
        # Convert to RGB (YOLO expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect potholes in the input frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (detections, annotated_frame)
            detections: List of dictionaries containing detection info
            annotated_frame: Frame with detection visualizations
        """
        if frame is None or frame.size == 0:
            return [], frame
        
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        
        try:
            # Run inference
            results = self.model(processed_frame, verbose=False)
            
            # Process detections
            detections = []
            annotated_frame = frame.copy()
            
            for result in results:
                # Get bounding boxes, confidence scores, and class IDs
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Filter for pothole class (class 0 in YOLO pothole models)
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, class_ids)):
                    if cls_id == 0:  # Assuming class 0 is pothole
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Add to detections
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(cls_id),
                            'class_name': 'pothole'
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'Pothole {conf:.2f}'
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return detections, annotated_frame
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return [], frame
    
    def detect_potholes(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Alias for detect() for backward compatibility"""
        return self.detect(frame)

# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize detector
    detector = YOLOv8PotholeDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect potholes
        detections, output_frame = detector.detect(frame)
        
        # Display results
        cv2.imshow('Pothole Detection', output_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
