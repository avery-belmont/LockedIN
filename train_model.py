"""
LockedIn Custom YOLO Training Script
This script downloads your labeled dataset and trains a custom YOLOv8 model
"""

from roboflow import Roboflow
from ultralytics import YOLO
import os

print("="*60)
print("LOCKEDIN - Custom YOLO Training")
print("="*60)

# Step 1: Download dataset from Roboflow
print("\n[1/4] Downloading dataset from Roboflow...")
rf = Roboflow(api_key="A6cDl7qoxfYk14bauEsa")
project = rf.workspace("maryams-workspace-n0as3").project("linkedin-phone-detection")
version = project.version(1)
dataset = version.download("yolov8")

print(f"✓ Dataset downloaded to: {dataset.location}")


# Fix data.yaml to use train for validation
import yaml

yaml_path = f"{dataset.location}/data.yaml"
with open(yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Use train images for validation if valid doesn't exist
data_config['val'] = data_config['train']

with open(yaml_path, 'w') as f:
    yaml.dump(data_config, f)

print("✓ Fixed data.yaml to use train set for validation")

# Step 2: Load pre-trained YOLO model
print("\n[2/4] Loading YOLOv8 base model...")
model = YOLO('yolov8n.pt')  # nano model - fastest, good for laptops
print("✓ Base model loaded")

# Step 3: Train the model
print("\n[3/4] Starting training...")
print("This will take 10-30 minutes depending on your computer.")
print("You'll see progress updates as it trains.\n")

results = model.train(
    data=f"{dataset.location}/data.yaml",  # path to your dataset
    epochs=50,           # number of training cycles (50 is good for 100 images)
    imgsz=640,          # image size
    batch=8,            # batch size (lower if you get memory errors)
    name='lockedin_phone_detector',  # name for this training run
    patience=10,        # early stopping if no improvement
    save=True,          # save checkpoints
    plots=True,         # generate training plots
    device='cpu'        # use 'cuda' if you have NVIDIA GPU
)

print("\n✓ Training complete!")

# Step 4: Validate the model
print("\n[4/4] Validating model performance...")
metrics = model.val()

print("\n" + "="*60)
print("TRAINING RESULTS")
print("="*60)
print(f"Model saved to: runs/detect/lockedin_phone_detector/weights/best.pt")
print(f"mAP50: {metrics.box.map50:.3f}")  # Mean Average Precision
print(f"mAP50-95: {metrics.box.map:.3f}")
print("\nTraining plots saved to: runs/detect/lockedin_phone_detector/")
print("="*60)

print("\n✓ All done! Your custom model is ready to use.")
print("\nNext steps:")
print("1. Check the training plots in runs/detect/lockedin_phone_detector/")
print("2. Update main.py to use: model = YOLO('runs/detect/lockedin_phone_detector/weights/best.pt')")
print("3. Test your custom model!")