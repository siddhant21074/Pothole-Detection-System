import os
import yaml
from ultralytics import YOLO
import shutil
from pathlib import Path

class PotholeDetectorTrainer:
    def __init__(self, data_dir='data/pothole_dataset', model_size='s'):
        """
        Initialize the pothole detector trainer
        
        Args:
            data_dir (str): Directory containing the dataset
            model_size (str): Model size ('s' for small, 'm' for medium, etc.)
        """
        self.data_dir = Path(data_dir)
        self.model_size = model_size
        self.model = None
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        (self.data_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
    def check_dataset_exists(self):
        """Check if the dataset exists and has the required structure"""
        required_dirs = ['images', 'labels']
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                print(f"⚠️ Directory not found: {dir_path}")
                return False
                
        # Check if there are any images and labels
        num_images = len(list((self.data_dir / 'images').glob('*')))
        num_labels = len(list((self.data_dir / 'labels').glob('*')))
        
        if num_images == 0 or num_labels == 0:
            print(f"⚠️ No images or labels found in {self.data_dir}")
            print(f"   Found {num_images} images and {num_labels} labels")
            return False
            
        print(f"✅ Found {num_images} images and {num_labels} labels in {self.data_dir}")
        return True
        
    def prepare_dataset(self, images_dir, labels_dir):
        """
        Prepare the dataset in YOLO format
        
        Args:
            images_dir: Path to directory containing images
            labels_dir: Path to directory containing YOLO format labels
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"Images or labels directory not found. Check paths:\n"
                                  f"Images: {images_dir}\n"
                                  f"Labels: {labels_dir}")

        # Copy images
        for img_file in images_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_file, self.data_dir / 'images' / img_file.name)
                print(f"📁 Copied {img_file.name} to {self.data_dir / 'images'}")
        
        # Copy labels
        for label_file in labels_dir.glob('*'):
            if label_file.suffix.lower() == '.txt':
                shutil.copy2(label_file, self.data_dir / 'labels' / label_file.name)
                print(f"📄 Copied {label_file.name} to {self.data_dir / 'labels'}")
    
    def create_yaml(self, class_names):
        """Create YAML configuration file for YOLOv8"""
        data = {
            'path': str(self.data_dir.absolute()),
            'train': 'images',
            'val': 'images',  # In a real scenario, use a separate validation set
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names)
        }
        
        yaml_path = self.data_dir / 'pothole_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
            
        print(f"📝 Created dataset config at: {yaml_path}")
        return yaml_path
    
    def train(self, epochs=50, imgsz=640, batch=16, lr0=0.01):
        """
        Train the YOLOv8 model
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            lr0: Initial learning rate
        """
        # Verify dataset exists before starting training
        if not self.check_dataset_exists():
            print("\n❌ Dataset not found or incomplete. Please prepare your dataset first.")
            print("Here's how to prepare your dataset:")
            print("\n1. Create the following directory structure:")
            print(f"   {self.data_dir}/")
            print("   ├── images/     # Put your training images here (.jpg, .png)")
            print("   └── labels/     # Put your YOLO-format labels here (.txt)")
            print("\n2. Each image should have a corresponding .txt file with the same name")
            print("   Example: 'image1.jpg' -> 'labels/image1.txt'")
            print("\n3. Label format (YOLO format):")
            print("   class x_center y_center width height")
            print("   (all values should be normalized to [0,1])")
            print("\n4. After preparing your dataset, run the script again.")
            return None
            
        print("\n🚀 Starting training...")
        
        # Load a pretrained YOLOv8 model
        self.model = YOLO(f'yolov8{self.model_size}.pt')
        
        # Train the model
        results = self.model.train(
            data=str(self.data_dir / 'pothole_dataset.yaml'),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            device='0' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu',
            project='pothole_detection',
            name=f'yolov8{self.model_size}_pothole',
            exist_ok=True
        )
        
        return results
def main():
    # Initialize trainer
    trainer = PotholeDetectorTrainer(data_dir='data/pothole_dataset', model_size='s')
    
    # Prepare dataset from raw data
    raw_data_dir = Path('data/raw/PUBLIC POTHOLE DATASET')
    
    # For training, we'll use the training split
    train_images_dir = raw_data_dir / 'train' / 'images'
    train_labels_dir = raw_data_dir / 'train' / 'labels'
    
    # Copy training data
    if train_images_dir.exists() and train_labels_dir.exists():
        print("📂 Found training data in raw directory")
        trainer.prepare_dataset(
            images_dir=train_images_dir,
            labels_dir=train_labels_dir
        )
    else:
        # Fallback to using all images if train split not found
        print("⚠️ Train split not found, using all available images")
        trainer.prepare_dataset(
            images_dir=raw_data_dir / 'images',
            labels_dir=raw_data_dir / 'labels'
        )
    
    # Create YAML configuration
    yaml_path = trainer.create_yaml(class_names=['pothole'])
    
    # Start training
    results = trainer.train(epochs=50, batch=8)
    
    if results:
        # Save the best model
        best_model_path = 'pothole_detection.pt'
        trainer.model.export(format='onnx')
        print(f"\n✅ Training completed! Model saved to: {best_model_path}")

if __name__ == "__main__":
    main()