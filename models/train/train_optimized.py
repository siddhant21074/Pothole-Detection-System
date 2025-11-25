# models/train/train_optimized.py
from ultralytics import YOLO
import os
import yaml
import shutil
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class OptimizedPotholeTrainer:
    def __init__(self, data_dir='data/raw/PUBLIC POTHOLE DATASET'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path('models/optimized')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def prepare_dataset(self, test_size=0.15, val_size=0.15):
        """Prepare dataset with optimized splitting."""
        print("Preparing dataset...")
        
        # Create directories
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Get all images
        images = list((self.data_dir / 'images').glob('*.*'))
        images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # Split into train, val, test
        train_val, test = train_test_split(images, test_size=test_size, random_state=42)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
        
        # Function to copy files
        def copy_files(files, split):
            for img_path in tqdm(files, desc=f'Processing {split} set'):
                # Copy image
                dst_img = self.data_dir / split / 'images' / img_path.name
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
                
                # Copy corresponding label
                label_path = (self.data_dir / 'labels' / img_path.stem).with_suffix('.txt')
                if label_path.exists():
                    dst_label = self.data_dir / split / 'labels' / label_path.name
                    if not dst_label.exists():
                        shutil.copy2(label_path, dst_label)
        
        # Copy files to respective directories
        copy_files(train, 'train')
        copy_files(val, 'val')
        copy_files(test, 'test')
        
        print(f"Dataset prepared with {len(train)} training, {len(val)} validation, and {len(test)} test samples.")
        return len(train), len(val), len(test)

    def create_yaml(self):
        """Create YAML configuration file for YOLO training."""
        yaml_content = {
            'path': str(self.data_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {0: 'pothole'},
            'nc': 1
        }
        
        yaml_path = self.output_dir / 'pothole_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
            
        return yaml_path

    def train(self):
        """Train an optimized YOLOv8 model for pothole detection."""
        # Prepare dataset
        train_size, val_size, test_size = self.prepare_dataset()
        yaml_path = self.create_yaml()
        
        # Calculate batch size based on available memory
        batch_size = 32 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9 else 16
        
        # Load YOLOv8s model (smaller and faster than YOLOv8n but still accurate)
        model = YOLO('yolov8s.pt')
        
        # Training configuration
        config = {
            'data': str(yaml_path),
            'epochs': 100,
            'imgsz': 640,
            'batch': batch_size,
            'workers': min(os.cpu_count(), 8),
            'device': '0' if torch.cuda.is_available() else 'cpu',
            'optimizer': 'AdamW',  # Better than SGD for this task
            'lr0': 0.01,           # Initial learning rate
            'lrf': 0.01,           # Final learning rate (lr0 * lrf)
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 0.05,           # Box loss gain
            'cls': 0.5,            # Class loss gain
            'dfl': 1.5,            # Distribution Focal Loss gain
            'hsv_h': 0.015,        # Image HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,          # Image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,          # Image HSV-Value augmentation (fraction)
            'degrees': 5.0,        # Image rotation (+/- deg)
            'translate': 0.1,      # Image translation (+/- fraction)
            'scale': 0.2,          # Image scale (+/- gain)
            'shear': 2.0,          # Image shear (+/- deg)
            'perspective': 0.0005, # Image perspective
            'flipud': 0.2,         # Image flip up-down (probability)
            'fliplr': 0.5,         # Image flip left-right (probability)
            'mosaic': 0.8,         # Image mosaic (probability)
            'mixup': 0.1,          # Image mixup (probability)
            'copy_paste': 0.2,     # Segment copy-paste (probability)
            'close_mosaic': 15,    # Disable mosaic for last N epochs
            'patience': 25,        # Early stopping patience
            'amp': True,           # Automatic Mixed Precision (AMP) training
            'cos_lr': True,        # Use cosine learning rate scheduler
            'project': str(self.output_dir),
            'name': 'yolov8s_pothole_optimized',
            'exist_ok': True,
        }
        
        # Train the model
        print("Starting training...")
        results = model.train(**config)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        metrics = model.val()
        
        # Print results
        print("\nTest Results:")
        print(f"- mAP50-95: {metrics.box.map:.4f}")
        print(f"- mAP50: {metrics.box.map50:.4f}")
        print(f"- Precision: {metrics.box.precision.mean():.4f}")
        print(f"- Recall: {metrics.box.recall.mean():.4f}")
        
        # Export to ONNX format for deployment
        print("\nExporting model to ONNX format...")
        model.export(
            format='onnx',
            dynamic=True,
            simplify=True,
            opset=12,
            imgsz=640,
            device='cpu' if not torch.cuda.is_available() else 0
        )
        
        print(f"\nTraining completed! Model saved to: {self.output_dir}")
        return results, metrics

if __name__ == "__main__":
    trainer = OptimizedPotholeTrainer()
    trainer.train()