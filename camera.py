from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
model = YOLO("/home/joao/Documents/GitHub/yolo/runs/detect/train11/weights/best.pt")

# Open the camera
cap = cv2.VideoCapture(2)  # 0 represents the default camera

# Set the desired frame width and height (optional, depends on your camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the codec and create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))  # 20 FPS, resolution 640x480

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Resize the frame to a smaller size for better performance
        resized_frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

        # Perform object detection on the resized frame
        results = model(resized_frame)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the camera and video writer, and close the window
cap.release()
out.release()
cv2.destroyAllWindows()