from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("/home/joao/Documents/GitHub/yolo/Elec_Compo_ide/runs/detect/train13/weights/best.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(
    data="/home/joao/Documents/GitHub/yolo/Elec_Compo_ide/train_fine_tuningV3/I2M.v6i.yolov8/data.yaml",  # Path to fine-tuning dataset
    epochs=600,  # Number of epochs for fine-tuning
    imgsz=480,  # Image size
    batch=8,
    freeze=10,  # Freeze the first 10 layers
    lr0=1e-4,
    verbose=True,
    # resume=True,
    plots=True
)