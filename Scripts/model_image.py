from ultralytics import YOLO
import cv2

# Load the model
model = YOLO(r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\runs\detect\train19\weights\best.pt")

# Load your image
image_path = r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\resultsandimages\20250630_132704.jpg"
image = cv2.imread(image_path)

# Optionally resize if needed
image = cv2.resize(image, (800, 800))

# Run YOLO detection
results = model(image)

# Annotate the image
annotated_image = results[0].plot()

# Show the result
cv2.imshow("YOLOv8 Object Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the result
cv2.imwrite("annotated_image.jpg", annotated_image)