from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\runs\detect\train8\weights\best.pt")

    results = model.train(
        data=r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\train_fine_tuningV6\data.yaml",
        epochs=600,
        imgsz=480,
        batch=8,
        freeze=10,
        lr0=0.0001,
        verbose=True,
        # resume=True,
        plots=True,
        device=0  # Use GPU
    )

if __name__ == '__main__':
    main()