from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\runs\detect\train23\weights\best.pt")

    results = model.train(
        data=r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\train_fine_tuningV7\data.yaml",
        epochs=100,
        imgsz=480,
        batch=8,
        freeze=10,
        lr0=0.000001,
        verbose=True,
        # resume=True,
        plots=True,
        device=0  # Use GPU
    )

if __name__ == '__main__':
    main()