from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8s.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(
    data="/home/joao/Documents/GitHub/yolo/train_fine_tuning/I2M.v3i.yolov8/data.yaml",  # Path to fine-tuning dataset
    epochs=300,  # Number of epochs for fine-tuning
    imgsz=480,  # Image size
    batch=8,
    # freeze=10,  # Freeze the first 10 layers
    # lr0=1e-4,
    verbose=True,
    # resume=True,
    plots=True
)