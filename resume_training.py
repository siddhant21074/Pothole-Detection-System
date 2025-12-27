import os
import torch
from ultralytics import YOLO
from pathlib import Path

def main():
    # Set device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    base_dir = Path('f:/Projects/Pothole Detection System')
    data_yaml = base_dir / 'models/optimized/pothole_dataset.yaml'
    
    # Use the YOLOv8m model as starting point
    model_weights = base_dir / 'yolov8m.pt'
    
    # Verify files exist
    if not model_weights.exists():
        print(f"Model weights not found at {model_weights}")
        return
    if not data_yaml.exists():
        print(f"Dataset YAML not found at {data_yaml}")
        return
    
    print("Loading model...")
    model = YOLO(str(model_weights))
    
    # Training configuration
    results = model.train(
        data=str(data_yaml),
        epochs=50,  # Total epochs (11 already completed, training 39 more)
        imgsz=640,
        batch=4,  # Reduced batch size for YOLOv8m on CPU
        device=0 if device == 'cuda' else 'cpu',
        workers=4,
        cache='ram',  # Use RAM caching for faster training
        optimizer='AdamW',
        lr0=0.001,  # Slightly higher learning rate for YOLOv8m
        resume=True,  # Important: This resumes training
        project=str(base_dir / 'pothole_detection'),
        name='yolov8m_pothole_resumed',
        exist_ok=True,
        close_mosaic=10,  # Disable mosaic augmentation in last 10 epochs
        hsv_h=0.015,     # Image HSV-Hue augmentation (fraction)
        hsv_s=0.7,       # Image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,       # Image HSV-Value augmentation (fraction)
        flipud=0.2,      # Image flip up-down (probability)
        fliplr=0.5,      # Image flip left-right (probability)
        mosaic=0.8,      # Image mosaic (probability)
        mixup=0.1,       # Image mixup (probability)
        copy_paste=0.2,  # Segment copy-paste (probability)
    )
    
    # Export to ONNX after training
    model.export(format='onnx', dynamic=True, simplify=True)
    print("Training completed and model exported to ONNX format.")

if __name__ == "__main__":
    main()
