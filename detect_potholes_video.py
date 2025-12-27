import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

def detect_potholes_in_video(video_path: str, output_path: str = 'output_detection.mp4', model_path: str = None, conf_threshold: float = 0.4):
    """
    Detect potholes in a video file and save the output using a custom-trained model.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the output video
        model_path: Path to the custom YOLOv8 model
        conf_threshold: Confidence threshold for detections (0-1)
    """
    # Load the custom model
    if model_path is None:
        model_path = 'models/optimized/yolov8s_pothole_optimized/weights/best.pt'
    
    print(f"Loading custom model from: {model_path}")
    model = YOLO(model_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.overrides['conf'] = conf_threshold
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = True
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect potholes in the frame
        results = model(frame, verbose=False)
        
        # Process detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes in (x1, y1, x2, y2) format
            confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                # Only process pothole class (class_id 0)
                if class_id != 0:  # Skip if not a pothole
                    continue
                    
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence
                label = f"Pothole: {conf:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Write frame to output video
        out.write(frame)
        
        # Display progress
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)", end='\r')
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nDetection complete! Output saved to: {output_path}")

if __name__ == "__main__":
    # Path to your video file
    video_path = "data/detect_video.mp4"
    
    # Output path for the processed video
    output_path = "data/detected_potholes_custom_model.mp4"
    
    # Path to custom model (None will use the default path)
    custom_model_path = None  # or specify the full path if different
    
    # Run detection with custom model
    detect_potholes_in_video(
        video_path=video_path,
        output_path=output_path,
        model_path=custom_model_path,
        conf_threshold=0.4  # Adjust confidence threshold as needed
    )
