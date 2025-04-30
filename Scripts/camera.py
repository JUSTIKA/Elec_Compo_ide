from ultralytics import YOLO
import cv2

# model = YOLO("/home/joao/Documents/GitHub/yolo/Elec_Compo_ide/runs/detect/train11/weights/best.pt") #model 11
model = YOLO("/home/joao/Documents/GitHub/yolo/Elec_Compo_ide/Test sets/train8/weights/best.pt")

cap = cv2.VideoCapture(0)  # 0 represents the default camera

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame from the camera.")
        break

    resized_frame = cv2.resize(frame, (640, 480))

    results = model(resized_frame)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()