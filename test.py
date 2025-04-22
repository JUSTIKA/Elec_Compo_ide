from ultralytics import YOLO

def main():
    # Load a COCO-pretrained YOLOv8n model
    model = YOLO(r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\runs\detect\train8\weights\best.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(
        data=r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\train_fine_tuningV6\data.yaml",  # Path to fine-tuning dataset
        epochs=600,  # Number of epochs for fine-tuning
        imgsz=480,  # Image size
        batch=8,
        freeze=10,  # Freeze the first 10 layers
        lr0=0.0001,
        verbose=True,
        # resume=True,
        plots=True,
        device=0  # Use GPU
    )

if __name__ == '__main__':
    main()